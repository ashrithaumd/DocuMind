"""The bank's customer data, under per-customer row-level security.

Owned tables (customers, accounts, cards, transactions, fees, holdings):
every row carries customer_id, and one policy per table, for every command,
lets a transaction see or write a row only when
    customer_id = app.customer_id
(set per transaction by db.access.set_access, like app.org_id). No context,
no rows; staff and admins have no customer_id, so they see no customer data.
FORCE ROW LEVEL SECURITY applies it to the table owner too: a loader that
owns the tables but can't bypass RLS (a Neon owner role without BYPASSRLS)
writes each customer's rows in a transaction bound to that customer.

A child row can't claim a different owner than its parent: accounts are
unique on (account_id, customer_id), and cards, transactions, fees and
holdings reference that pair (transactions and fees also reference their
card and transaction by (id, customer_id)).

Reference data (securities, prices) is the same for everyone: no RLS.

users.customer_id links a signed-in user to at most one customer.

The app role reads only: tools read customer data, nothing in the app
writes it. Money is integer cents; prices are numeric.

Revision ID: 0009
Revises: 0008
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

APP_ROLE = "finpilot_app"
OWNED = ("customers", "accounts", "cards", "transactions", "fees", "holdings")
REFERENCE = ("securities", "prices")
POLICY = "customer_id = nullif(current_setting('app.customer_id', true), '')"

TABLES = """
CREATE TABLE customers (
    customer_id text PRIMARY KEY,
    org_id      text NOT NULL REFERENCES orgs,
    full_name   text NOT NULL,
    segment     text NOT NULL CHECK (segment IN ('retail', 'premier', 'student')),
    joined_on   date NOT NULL
);

CREATE TABLE accounts (
    account_id  text PRIMARY KEY,
    customer_id text NOT NULL REFERENCES customers,
    type        text NOT NULL CHECK (type IN ('checking', 'savings', 'credit', 'brokerage')),
    product     text NOT NULL,
    opened_on   date NOT NULL,
    status      text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'frozen', 'closed')),
    UNIQUE (account_id, customer_id)
);
CREATE INDEX ix_accounts_customer ON accounts (customer_id);

CREATE TABLE cards (
    card_id     text PRIMARY KEY,
    account_id  text NOT NULL,
    customer_id text NOT NULL,
    last4       text NOT NULL CHECK (last4 ~ '^[0-9]{4}$'),
    network     text NOT NULL CHECK (network IN ('visa', 'mastercard')),
    status      text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'locked', 'closed')),
    expires_on  date NOT NULL,
    UNIQUE (card_id, customer_id),
    FOREIGN KEY (account_id, customer_id) REFERENCES accounts (account_id, customer_id)
);
CREATE INDEX ix_cards_customer ON cards (customer_id);

CREATE TABLE transactions (
    txn_id       text PRIMARY KEY,
    account_id   text NOT NULL,
    customer_id  text NOT NULL,
    card_id      text,
    posted_on    date NOT NULL,
    amount_cents bigint NOT NULL CHECK (amount_cents <> 0),  -- negative = money out
    merchant     text,
    category     text NOT NULL,
    description  text NOT NULL,
    UNIQUE (txn_id, customer_id),
    FOREIGN KEY (account_id, customer_id) REFERENCES accounts (account_id, customer_id),
    FOREIGN KEY (card_id, customer_id) REFERENCES cards (card_id, customer_id)
);
CREATE INDEX ix_transactions_customer_posted ON transactions (customer_id, posted_on);
CREATE INDEX ix_transactions_account_posted ON transactions (account_id, posted_on);

CREATE TABLE fees (
    fee_id       text PRIMARY KEY,
    account_id   text NOT NULL,
    customer_id  text NOT NULL,
    txn_id       text,
    fee_type     text NOT NULL CHECK (fee_type IN ('overdraft', 'nsf', 'monthly_maintenance',
                                                   'atm', 'foreign_transaction', 'wire',
                                                   'late_payment')),
    amount_cents bigint NOT NULL CHECK (amount_cents > 0),
    assessed_on  date NOT NULL,
    waived       boolean NOT NULL DEFAULT false,
    FOREIGN KEY (account_id, customer_id) REFERENCES accounts (account_id, customer_id),
    FOREIGN KEY (txn_id, customer_id) REFERENCES transactions (txn_id, customer_id)
);
CREATE INDEX ix_fees_customer_assessed ON fees (customer_id, assessed_on);

CREATE TABLE securities (
    symbol        text PRIMARY KEY,
    name          text NOT NULL,
    type          text NOT NULL CHECK (type IN ('etf', 'stock', 'mutual_fund')),
    expense_ratio numeric(6, 4) CHECK (expense_ratio >= 0)
);

CREATE TABLE prices (
    symbol     text NOT NULL REFERENCES securities,
    price_date date NOT NULL,
    close      numeric(14, 4) NOT NULL CHECK (close > 0),
    source     text NOT NULL,
    PRIMARY KEY (symbol, price_date)
);

CREATE TABLE holdings (
    account_id       text NOT NULL,
    customer_id      text NOT NULL,
    symbol           text NOT NULL REFERENCES securities,
    quantity         numeric(18, 6) NOT NULL CHECK (quantity > 0),
    cost_basis_cents bigint NOT NULL CHECK (cost_basis_cents >= 0),
    PRIMARY KEY (account_id, symbol),
    FOREIGN KEY (account_id, customer_id) REFERENCES accounts (account_id, customer_id)
);
CREATE INDEX ix_holdings_customer ON holdings (customer_id);

ALTER TABLE users
    ADD COLUMN customer_id text UNIQUE REFERENCES customers;
"""


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(TABLES)
    for table in OWNED:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY customer_owner ON {table} USING ({POLICY}) WITH CHECK ({POLICY})")
    op.execute(f"GRANT SELECT ON {', '.join(OWNED + REFERENCE)} TO {APP_ROLE}")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE users DROP COLUMN customer_id")
    for table in ("holdings", "prices", "securities", "fees", "transactions", "cards",
                  "accounts", "customers"):
        op.execute(f"DROP TABLE {table}")

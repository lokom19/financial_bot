# import psycopg2  # Add this for PostgreSQL connection
#
# DB_HOST = "localhost"  # Change this to your database host
# DB_PORT = "5432"  # Default PostgreSQL port
# DB_NAME = "postgres"  # Change to your database name
# DB_USER = "postgres"  # Change to your username
# DB_PASSWORD = "mysecretpassword"  # Change to your password
#
#
# def setup_database():
#     """Set up the database schema and table if they don't exist"""
#     try:
#         # Connect to the database
#         conn = psycopg2.connect(
#             host=DB_HOST,
#             port=DB_PORT,
#             dbname=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD
#         )
#
#         # Create a cursor
#         cursor = conn.cursor()
#
#         # Create the table if it doesn't exist
#         create_table_query = """
#         CREATE TABLE IF NOT EXISTS public.model_results (
#             id SERIAL PRIMARY KEY,
#             db_name VARCHAR(255) NOT NULL,
#             model_name VARCHAR(255) NOT NULL,
#             timestamp TIMESTAMP NOT NULL,
#             text TEXT NOT NULL
#         );
#         """
#
#         cursor.execute(create_table_query)
#         conn.commit()
#
#         print("Database table 'public.model_results' is ready")
#
#         # Close the connection
#         cursor.close()
#         conn.close()
#
#         return True
#
#     except Exception as e:
#         print(f"Error setting up database: {e}")
#         return False
#
#
# setup_database()


import psycopg2  # Add this for PostgreSQL connection

DB_HOST = "46.252.251.117"  # Change this to your database host
DB_PORT = "4791"  # Default PostgreSQL port
DB_NAME = "postgres"  # Change to your database name
DB_USER = "postgres"  # Change to your username
DB_PASSWORD = "mysecretpassword"  # Change to your password


def setup_database():
    """Set up the database schema and table if they don't exist"""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Create a cursor
        cursor = conn.cursor()

        # Create the schema if it doesn't exist
        create_schema_query = """
        CREATE SCHEMA IF NOT EXISTS crypto;
        """
        cursor.execute(create_schema_query)
        conn.commit()

        print("Schema 'crypto' created or already exists")

        # Create the table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS crypto.ticker_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(50) NOT NULL,
            timestamp BIGINT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            last NUMERIC(20, 8) NOT NULL,
            bid NUMERIC(20, 8),
            ask NUMERIC(20, 8),
            high NUMERIC(20, 8),
            low NUMERIC(20, 8),
            open NUMERIC(20, 8),
            close NUMERIC(20, 8),
            vwap NUMERIC(20, 8),
            change NUMERIC(20, 8),
            percentage NUMERIC(10, 4),
            base_volume NUMERIC(25, 12),
            quote_volume NUMERIC(25, 12),
            info JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor.execute(create_table_query)

        # Создаем индекс для быстрого поиска по symbol и timestamp
        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_ticker_data_symbol_timestamp 
        ON crypto.ticker_data(symbol, timestamp);
        """
        cursor.execute(create_index_query)

        conn.commit()

        print("Database table 'trading.ticker_data' is ready")

        # Close the connection
        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"Error setting up database: {e}")
        return False


setup_database()
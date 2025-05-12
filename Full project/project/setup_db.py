import sqlite3

def create_database():
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect("feedback.db")

    # Create the feedback table if it doesn't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL
        )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Database setup complete. Table 'feedback' is ready.")

if __name__ == "__main__":
    create_database()

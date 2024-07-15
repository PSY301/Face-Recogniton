import sqlite3
from functools import wraps


class DBWorker:
    """ This is a class to work with databases. """

    def __init__(self, path: str):
        """
        Constructor to initialize the attributes of the class.

        :param:path[str]
            absolute path to database.
        """
        self.path = path
        self.cursor = None

    def connector(self, func):
        """
        Decorator that helps with writing functions in this class.
        establishes a connection to the database and commits changes to it.

        :param:func
            function in the class we are working with.

        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            conn = sqlite3.connect(self.path)
            self.cursor = conn.cursor()
            result = getattr(DBWorker, func)(self, *args, **kwargs)
            conn.commit()
            conn.close()
            return result
        return wrapper

    def create(self):
        """
        Creates a database in which it creates a Users table with parameters(id[int], name[str], is_aug[bool]).
        """
        self.cursor.execute('''
                        CREATE TABLE IF NOT EXISTS Users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL
                        )
                        ''')

    def new_row(self, name: str) -> int:
        """
        Creates a new row in the Users table. Returns new user id.

        :param: name[str]
            name of user.
        :return:int
        """
        self.cursor.execute("SELECT MAX(id) FROM Users")
        result = self.cursor.fetchone()
        max_id = result[0] if result[0] else 0
        self.cursor.execute('INSERT INTO Users (id, name) VALUES (?,?)',
                            (max_id + 1, name))
        return max_id + 1

    def get_profile(self, user_id: int) -> str:
        """
        Returns the name of the user in the Users table whose id is person_id.

        :param: user_id[int]
            id of user.
        :return:str.
        """
        profile = 'Unknown Person'
        self.cursor.execute('SELECT name FROM Users WHERE id=?', (int(user_id),))
        results = self.cursor.fetchall()
        for row in results:
            profile = row
        return profile

    def get_names(self):
        """
        Returns all names in the Users table.

        :return:list
        """
        self.cursor.execute('SELECT name FROM Users')
        results = self.cursor.fetchall()
        for i, tp in enumerate(results):
            results[i] = tp[0]
        return results

    def get_id(self, name: str) -> int:
        """
        Returns the id of the user in the Users table whose name is name.

        :param: name[str]
            name of user.
        :return:int.
        """
        self.cursor.execute('SELECT id FROM Users WHERE name=?', (name,))
        results = self.cursor.fetchall()
        return results[0][0]

    def __getattribute__(self, item):
        """
        decorates each function in the class with a connector decorator.

        :param:item
            function in this class.
        """
        if item != "connector" and item in DBWorker.__dict__:
            return DBWorker.connector(self, item)
        else:
            return object.__getattribute__(self, item)

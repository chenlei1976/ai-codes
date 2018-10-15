
# -*- coding: UTF-8 -*-

import configparser
import pymssql
# import MySQLdb


class DBHelper(object):

    def __init__(self):
        files = ['sql_server.ini']
        cfg = configparser.ConfigParser()
        dataset = cfg.read(files)
        if len(dataset) != len(files):
            raise ValueError("Failed to open/find all files")
        # cfg.sections()
        try:
            self.host = cfg.get('db', 'host')
            self.port = cfg.get('db', 'port')
            self.user = cfg.get('db', 'user')
            self.passwd = cfg.get('db', 'passwd')
            self.database = cfg.get('db', 'database')
        except configparser.NoSectionError:
            raise ValueError("No section[db] in ini files")

    def connectDatabase(self):
        # conn = MySQLdb.connect(host=self.host,
        #                        port=self.port,
        #                        user=self.user,
        #                        passwd=self.passwd,
        #                        db=self.database,
        #                        charset='utf8')  # for chinese
        conn = pymssql.connect(host=self.host,
                               port=self.port,
                               user=self.user,
                               passwd=self.passwd,
                               database=self.database,
                               charset='utf8')  # for chinese
        return conn

    def createDatabase(self):

        conn = self.connectDatabase()  # connect Database

        sql = "create database if not exists " + self.database
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def createTable(self, sql):
        conn = self.connectDatabase()

        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def insert(self, sql, *params):
        conn = self.connectDatabase()

        cur = conn.cursor();
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def update(self, sql, *params):
        conn = self.connectDatabase()

        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def delete(self, sql, *params):
        conn = self.connectDatabase()

        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def select(self, sql, *params):
        conn = self.connectDatabase()

        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        while row:
            # print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]))
            row = cur.fetchone()
        cur.close()
        conn.close()


class TestDBHelper(object):
    def __init__(self):
        self.dbHelper = DBHelper()

    def testCreateDatebase(self):
        self.dbHelper.createDatabase()

    def testCreateTable(self):
        sql = "create table testtable(id int primary key auto_increment,name varchar(50),url varchar(200))"
        self.dbHelper.createTable(sql)

    def testInsert(self):
        sql = "insert into testtable(name,url) values(%s,%s)"
        params = ("test", "test")
        self.dbHelper.insert(sql, *params)

    def testUpdate(self):
        sql = "update testtable set name=%s,url=%s where id=%s"
        params = ("update", "update", "1")
        self.dbHelper.update(sql, *params)

    def testDelete(self):
        sql = "delete from testtable where id=%s"
        params = ("1")
        self.dbHelper.delete(sql, *params)


if __name__ == "__main__":
    testDBHelper = TestDBHelper()
    # testDBHelper.testCreateDatebase()
    # testDBHelper.testCreateTable()
    # testDBHelper.testInsert()
    # testDBHelper.testUpdate()
    # testDBHelper.testDelete()

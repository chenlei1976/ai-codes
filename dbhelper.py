
# -*- coding: UTF-8 -*-

import configparser
import logging
import pymssql
# import MySQLdb
import os
import pyodbc
import h5py
import tools

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
            self.password = cfg.get('db', 'password')
            self.database = cfg.get('db', 'database')
            self.timeout = cfg.get('db', 'timeout')
        except configparser.NoSectionError:
            raise ValueError("No section[db] in ini files")

    def connectDatabase(self):
        # TODO::Later use factory
        # conn = MySQLdb.connect(host=self.host,
        #                        port=self.port,
        #                        user=self.user,
        #                        passwd=self.password,
        #                        db=self.database,
        #                        charset='utf8')  # for chinese

        # conn = pymssql.connect(host=self.host,
        #                        port=self.port,
        #                        user=self.user,
        #                        password=self.password,
        #                        database=self.database,
        #                        charset='utf8')  # for chinese

        connInfo = 'DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s' % (
            self.host, self.database, self.user, self.password)

        try:
            conn = pyodbc.connect(connInfo, timeout=self.timeout)
        except pyodbc.Error as err:
            logging.error('can not connect[%s]!' % connInfo)
            raise Exception("Couldn't connect[{}]".format(connInfo))

        return conn

    def createDatabase(self):

        conn = self.connectDatabase()  # connect Database

        sql = "create database if not exists " + self.database

        try:
            cur = conn.cursor()
            cur.execute(sql)
        except pyodbc.Error as err:
            logging.error('execute create db [%s] failed!' % sql)
            raise Exception('execute create db [%s] failed!' % sql)
        finally:
            cur.close()
            conn.close()

    def createTable(self, sql):
        conn = self.connectDatabase()

        try:
            cur = conn.cursor()
            cur.execute(sql)
        except pyodbc.Error as err:
            logging.error('execute create table [%s] failed!' % sql)
            raise Exception('execute create table [%s] failed!' % sql)
        finally:
            cur.close()
            conn.close()

    def insert(self, sql, *params):
        conn = self.connectDatabase()

        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
        except pyodbc.Error as err:
            logging.error('execute insert [%s] failed!' % sql)
            raise Exception('execute insert [%s] failed!' % sql)
        finally:
            cur.close()
            conn.close()

    def update(self, sql, *params):
        conn = self.connectDatabase()

        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
        except pyodbc.Error as err:
            logging.error('execute update[%s] failed!' % sql)
            raise Exception('execute update[%s] failed!' % sql)
        finally:
            cur.close()
            conn.close()

    def delete(self, sql, *params):
        conn = self.connectDatabase()

        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
        except pyodbc.Error as err:
            logging.error('execute delete[%s] failed!' % sql)
            raise Exception('execute delete[%s] failed!' % sql)
        finally:
            cur.close()
            conn.close()

    def select(self, sql, *params):
        conn = self.connectDatabase()

        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            row = cur.fetchone()
            while row:
                print(','.join(map(str, row)))
                row = cur.fetchone()
        except pyodbc.Error as err:
            logging.error('execute select[%s] failed!' % sql)
            raise Exception('execute select[%s] failed!' % sql)
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def fetchsome(cursor, arraySize=1000):
        """ A generator that simplifies the use of fetchmany """
        while True:
            results = cursor.fetchmany(arraySize)
            if not results:
                break
            for result in results:
                yield result


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
    # testDBHelper = TestDBHelper()
    # testDBHelper.testCreateDatebase()
    # testDBHelper.testCreateTable()
    # testDBHelper.testInsert()
    # testDBHelper.testUpdate()
    # testDBHelper.testDelete()

    # dbhelper = DBHelper()
    # sql = 'SELECT * from dbevent where id<5'
    # dbhelper.select(sql)

    # Fetching Large Record Sets from a Database with a Generator

    columns = ['ClaimId', 'PolicyNumber', 'FormLocation', 'FilePath']

    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()

    cur = conn.cursor()
    sql = """
        SELECT C.ID AS ClaimId, C.PolicyNumber, C.CreatedDate, CT.FormLocation 
        FROM T_Claim C , T_Claim_Travel CT WHERE CT.ClaimID = C.Id
        """

    cur.execute(sql)

    for result in dbhelper.fetchsome(cur, arraysize=2):
        # filePath = result[columns.index('FilePath')]
        filePath = result.FilePath
        fileName = os.path.basename(filePath)

        os.path.join('root', result.FormLocation, fileName)

    cur.close()
    conn.close()

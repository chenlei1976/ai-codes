
# -*- coding: UTF-8 -*-

import configparser
import logging
import os

try:
    import pypyodbc as pyodbc
    print("using pypyodbc")
except ImportError:
    print("using pyodbc")
    import pyodbc

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
            self.timeout = int(cfg.get('db', 'timeout'))
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
    def fetchsome(cursor, arraySize=5000):
        """ A generator that simplifies the use of fetchmany """
        while True:
            results = cursor.fetchmany(arraySize)
            if not results:
                break
            for result in results:
                yield result

    @staticmethod
    def executemany(conn, sql, params):
        ret = True
        try:
            cur = conn.cursor()
            # print('params ', params)
            cur.executemany(sql, params)
            conn.commit()
        except Exception:
            logging.error('executemany [%s] failed!' % sql)
            # assert False
            # raise Exception('executemany [%s] failed!' % sql)
            ret = False
        finally:
            cur.close()
            return ret

    @staticmethod
    def insertRecord(conn, sql, *params):
        ret = True
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
        except Exception:
            logging.error('insert [%s] failed!' % sql)
            ret = False
        finally:
            cur.close()
            return ret

    @staticmethod
    def updateRecord(conn, sql, *params):
        ret = True
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            conn.commit()
        except Exception:
            logging.error('update [%s] failed!' % sql)
            ret = False
        finally:
            cur.close()
            return ret


    @staticmethod
    def getValue(conn, sql, *params):
        num = -1
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            results = cur.fetchone()
            if results[0] is None:
                num = 0
            else:
                num = int(results[0])
        except Exception:
            logging.error('query sql[%s] failed!' % sql)
            num = -1
        finally:
            cur.close()
            return num


if __name__ == "__main__":

    # test1
    # dbHelper = DBHelper()
    # sql = "SELECT ID, ClaimID, FilePath, UploadDate, UploadBy FROM online.dbo.T_Claim_Upload where id<3;"
    # dbHelper.select(sql)

    # test2
    # columns = ['ClaimId', 'PolicyNumber', 'FormLocation', 'FilePath']
    #
    # dbhelper = DBHelper()
    # conn = dbhelper.connectDatabase()
    #
    # cur = conn.cursor()
    # sql = """
    #     SELECT C.ID AS ClaimId, C.PolicyNumber, C.CreatedDate, CT.FormLocation
    #     FROM T_Claim C , T_Claim_Travel CT WHERE CT.ClaimID = C.Id
    #     """
    #
    # cur.execute(sql)
    #
    # for result in dbhelper.fetchsome(cur, arraysize=2):
    #     # filePath = result[columns.index('FilePath')]
    #     filePath = result.FilePath
    #     fileName = os.path.basename(filePath)
    #
    #     os.path.join('root', result.FormLocation, fileName)
    #
    # cur.close()
    # conn.close()

    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()

    cur = conn.cursor()

    sql = 'SELECT ID, ClaimID, FilePath FROM claim.dbo.T_Claim_Upload WHERE id<3'
    cur.execute(sql)

    rootPath = "I:\\05_Claims\\02_eClaim_Files"

    for result in dbhelper.fetchsome(cur, arraySize=2):
        # filePath = result[columns.index('FilePath')]
        # filePath = result.FilePath
        filePath = result[2]
        fileName = os.path.basename(filePath)
        print(','.join(map(str, result)))
        subFolder = str(filePath).split("\\\\")[1]

        possibleFile2 = os.path.join(rootPath, subFolder)

        subFolder = subFolder.split("\\")[0]
        # print('subFolder ', subFolder)

        possibleFile1 = os.path.join(rootPath, subFolder, fileName)
        # print('real path ', realPath)
    cur.close()
    conn.close()
    

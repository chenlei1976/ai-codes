
# -*- coding: UTF-8 -*-
import paramiko
import os
import sys
from stat import S_ISDIR as isdir


class FileHandler(object):

    def __init__(self):
        files = ['file_server.ini']
        cfg = configparser.ConfigParser()
        dataset = cfg.read(files)
        if len(dataset) != len(files):
            raise ValueError("Failed to open/find all files")
        # cfg.sections()
        try:
            self.host = cfg.get('server', 'host')
            self.port = cfg.get('server', 'port')
            self.user = cfg.get('server', 'user')
            self.password = cfg.get('server', 'password')
        except configparser.NoSectionError:
            raise ValueError("No section[db] in ini files")

    def connectSftp(self):
        client = None
        sftp = None
        try:
            client = paramiko.Transport((self.host, self.port))
        except Exception as error:
            print(error)
        else:
            try:
                client.connect(username=self.user, password=self.password)
            except Exception as error:
                print(error)
            else:
                sftp = paramiko.SFTPClient.from_transport(client)
        return client, sftp

    def disconnect(self, client):
        try:
            client.close()
        except Exception as error:
            print(error)

    def get(self, sftp, remote, local):
        def _check_local(local):
            if not os.path.exists(local):
                try:
                    os.mkdir(local)
                except IOError as err:
                    print(err)

        # check existing for remote file
        try:
            result = sftp.stat(remote)
        except IOError as err:
            error = '[ERROR %s] %s: %s' % (err.errno, os.path.basename(os.path.normpath(remote)), err.strerror)
            print(error)
        else:
            if isdir(result.st_mode):
                # is folder
                dirname = os.path.basename(os.path.normpath(remote))
                local = os.path.join(local, dirname)
                _check_local(local)
                for file in sftp.listdir(remote):
                    sub_remote = os.path.join(remote, file)
                    sub_remote = sub_remote.replace('\\', '/')
                    get(sftp, sub_remote, local)
            else:
                # is file & copy
                if os.path.isdir(local):
                    local = os.path.join(local, os.path.basename(remote))
                try:
                    sftp.get(remote, local)
                except IOError as err:
                    print(err)
                else:
                    print('[get]', local, '<==', remote)


if __name__ == "__main__":
    fileHandler = FileHandler()
    client, sftp = fileHandler.connectSftp()
    fileHandler.get(sftp,'remote/file','local/file')
    fileHandler.disconnect(client)



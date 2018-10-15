
# -*- coding: UTF-8 -*-
import paramiko
import os
import sys
from stat import S_ISDIR as isdir


# from paramiko import SSHClient
# from scp import SCPClient
#
# ssh = SSHClient()
# ssh.load_system_host_keys()
# ssh.connect('example.com')
#
# sftp = ssh.open_sftp()
# sftp.put(localpath, remotepath)
# sftp.close()
# ssh.close()

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

    def put(self, sftp, local, remote):
        # 检查路径是否存在
        def _is_exists(path, function):
            path = path.replace('\\', '/')
            try:
                function(path)
            except Exception as error:
                return False
            else:
                return True

        # 拷贝文件
        def _copy(sftp, local, remote):
            # 判断remote是否是目录
            if _is_exists(remote, function=sftp.chdir):
                # 是，获取local路径中的最后一个文件名拼接到remote中
                filename = os.path.basename(os.path.normpath(local))
                remote = os.path.join(remote, filename).replace('\\', '/')
            # 如果local为目录
            if os.path.isdir(local):
                # 在远程创建相应的目录
                _is_exists(remote, function=sftp.mkdir)
                # 遍历local
                for file in os.listdir(local):
                    # 取得file的全路径
                    localfile = os.path.join(local, file).replace('\\', '/')
                    # 深度递归_copy()
                    _copy(sftp=sftp, local=localfile, remote=remote)
            # 如果local为文件
            if os.path.isfile(local):
                try:
                    sftp.put(local, remote)
                except Exception as error:
                    print(error)
                    print('[put]', local, '==>', remote, 'FAILED')
                else:
                    print('[put]', local, '==>', remote, 'SUCCESSED')

        # 检查local
        if not _is_exists(local, function=os.stat):
            print("'" + local + "': No such file or directory in local")
            return False
        # 检查remote的父目录
        remote_parent = os.path.dirname(os.path.normpath(remote))
        if not _is_exists(remote_parent, function=sftp.chdir):
            print("'" + remote + "': No such file or directory in remote")
            return False
        # 拷贝文件
        _copy(sftp=sftp, local=local, remote=remote)


if __name__ == "__main__":
    fileHandler = FileHandler()
    client, sftp = fileHandler.connectSftp()
    fileHandler.get(sftp,'remote/file','local/file')
    fileHandler.disconnect(client)



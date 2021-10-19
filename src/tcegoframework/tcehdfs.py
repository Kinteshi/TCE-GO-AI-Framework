# TRIBUNAL DE CONTAS DO ESTADO DE GOIAS
# SECRETARIA DE CONTROLE EXTERNO
# SERVICO DE INFORMACOES ESTRATEGICAS

import time
from subprocess import PIPE, Popen

import pandas as pd
import pexpect
from hdfs3 import HDFileSystem


class TCEHDFS:

    def __init__(self, domain, url, port, user, password):

        try:
            self.domain = domain
            self.url = url
            self.port = port
            self.user = user
            self.pwd = password

            self.popen = Popen(
                ['kdestroy', '-A'],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE)
            self.child = pexpect.spawn(f'kinit {self.user}@{self.domain}')
            self.child.expect('Password*')
            self.child.sendline(str(self.pwd))

            # Esperar 3 segundos para fazer conexao (se nao esperar, emite erro de autenticacao no AD)
            time.sleep(10)
            self.hdfs = HDFileSystem(
                host=self.url,
                port=self.port,
                pars={"hadoop.security.authentication": "kerberos"},
                user=self.user)

        except Exception as e:
            print("Erro ao logar no HDFS. Erro:", e)

    def readHDFS(self, path, file):
        with self.hdfs.open(path + file) as k:
            df = pd.read_csv(k, sep=',', low_memory=False)
            k.close()

        return df

    def writeHDFS(self, path, file, df):

        with self.hdfs.open(path + file, 'wb') as f:
            df.to_csv(f, index=False)
            f.close()

        return path + file

    def copyToHDFS(self, path_local_file, file_name, hdfs_path):
        self.hdfs.put(path_local_file + file_name, hdfs_path + file_name)
        pass

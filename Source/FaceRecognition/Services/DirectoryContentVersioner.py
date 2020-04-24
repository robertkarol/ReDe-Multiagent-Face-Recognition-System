from Utils.Singleton import SingletonMeta
from os import listdir
from typing import List, Tuple
import re


class DirectoryContentVersioner(metaclass=SingletonMeta):
    def __init__(self):
        self.__version_regex = '.+-v([0-9]+)'
        self.__content_regex = '{}-v[0-9]+'
        self.__version_suffix = '-v{}'
        self.__versioning_key = lambda x: x[1]

    def get_versioned_content(self, directory_path: str, content_basename: str) -> List[Tuple[str, int]]:
        return [(content, int(re.search(self.__version_regex, content).group(1))) for content in listdir(directory_path)
                if re.match(self.__content_regex.format(content_basename), content)]

    def get_latest(self, directory_path: str, content_basename: str) -> Tuple[str, int]:
        content = self.get_versioned_content(directory_path, content_basename)
        return max(content, key=self.__versioning_key) if len(content) > 0 else (None, -1)

    def get_earliest(self, directory_path: str, content_basename: str) -> Tuple[str, int]:
        content = self.get_versioned_content(directory_path, content_basename)
        return min(content, key=self.__versioning_key) if len(content) > 0 else (None, -1)

    def get_specific_version(self, directory_path: str, content_basename: str, version: int):
        return [content for content in self.get_versioned_content(directory_path, content_basename)
                if self.__versioning_key(content) == version][0]

    def get_versioned_name(self, content_basename: str, version: int):
        return content_basename + self.__version_suffix.format(version)

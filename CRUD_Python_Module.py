# CRUD_Python_Module.py
# Grazioso Salvare CRUD class used by ProjectOneTestScript.ipynb

from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError

class AnimalShelter(object): 
    """ CRUD operations for animals collection in the aac database. """ 

    def __init__(self,
                 username: str,
                 password: str,
                 host: str = "localhost",
                 port: int = 27017,
                 auth_source: str = "admin",
                 db_name: str = "aac",
                 collection_name: str = "animals",
                ):
        
        # Connection Variables 
        USER = username 
        PASS = password 
        HOST = host
        PORT = port
        DB = db_name
        COL = collection_name
        AUTH_SOURCE = auth_source
        
        # Initialize connection and fail fast if credentials are wrong 
        try:
            uri = f"mongodb://{USER}:{PASS}@{HOST}:{PORT}/?authSource={AUTH_SOURCE}"
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000) 
            self.client.admin.command("ping")
        except Exception as exc:
            raise Exception(f"Could not connect to MongoDB. Details: {exc}")
            
        self.database = self.client[DB]
        self.collection = self.database[COL]

    # Create: insert a single document
    # Input: data is the key/value pairs to insert
    # Return: True if successful insert, else False
    def create(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or not data:
            return False
        try:
            result = self.collection.insert_one(data)
            return bool(result.acknowledged and result.inserted_id)
        except Exception:
            return False
        
    # Read: use find() and return a list (not find_one)
    # Input: query dict; optional projection dict; optional limit
    # Return: list of documents if successful, else []
    def read(self,
             query: Optional[Dict[str, Any]] = None,
             projection: Optional[Dict[str, int]] = None,
             limit: int = 0,
            ) -> List[Dict[str, Any]]:
        try:
            q = query or {}
            proj = projection or {}
            cursor = self.collection.find(q, proj)
            if isinstance(limit, int) and limit > 0:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception:
            return []
    
    # Update: query for and change documents(s)
    # Input: query dict, new_values dict accepted by update_one/update_many, many flag
    # Return: number of objects modified
    def update(
        self,
        query: Dict[str, Any],
        new_values: Dict[str, Any],
        many: bool = True,
    ) -> int:
        if not isinstance(query, dict) or not isinstance(new_values, dict):
            return 0
        try:
            if many:
                result = self.collection.update_many(query, new_values)
            else:
                result = self.collection.update_one(query, new_values)
            return int(getattr(result, "modified_count", 0))
        except PyMongoError:
            return 0
        
    # Delete: query for and remove document(s)
    # Input: query dict, many flag
    # Return: number of objects removed
    def delete(
        self,
        query: Dict[str, Any],
        many: bool = True,
    ) -> int:
        if not isinstance(query, dict) or not query:
            return 0
        try:
            if many:
                result = self.collection.delete_many(query)
            else:
                result = self.collection.delete_one(query)
            return int(getattr(result, "deleted_count", 0))
        except PyMongoError:
            return 0
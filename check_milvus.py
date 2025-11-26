from pymilvus import connections, utility, Collection
import socket

def check_port(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((host, port))
            if result == 0:
                print(f"Port {port} on {host} is OPEN.")
                return True
            else:
                print(f"Port {port} on {host} is CLOSED (code {result}).")
                return False
    except Exception as e:
        print(f"Error checking port: {e}")
        return False

def check_milvus():
    host = "127.0.0.1"
    port = "19530"
    
    if not check_port(host, int(port)):
        print("Aborting Milvus connection attempt.")
        return

    try:
        print(f"Connecting to Milvus at {host}:{port}...")
        connections.connect("default", host=host, port=port)
        print("Connected.")
        
        collections = utility.list_collections()
        print(f"Collections: {collections}")
        
        if "music_embeddings" in collections:
            collection = Collection("music_embeddings")
            print(f"Collection 'music_embeddings' loaded.")
            print(f"Number of entities: {collection.num_entities}")
            
            # Check if index exists
            print(f"Indexes: {collection.indexes}")
        else:
            print("Collection 'music_embeddings' NOT found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_milvus()

import cProfile
import pstats
import io
from backend.query.query_data import query_rag

def profile_query(query):
    pr = cProfile.Profile()
    pr.enable()
    
    response = query_rag(query)
    
    pr.disable()
    with open("profile_output.txt", "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats("cumulative")
        ps.print_stats()

if __name__ == "__main__":
    query = "what is creamobile?"
    profile_query(query)

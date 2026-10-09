import os
from kristy import KristyGraphMath

def main():
    db_path = "./data/knowledge_base.db"
    
    if not os.path.exists(db_path):
        print("CRITICAL: Database not found. System requires 'prepare.py' to be run.")
        return

    kristy = KristyGraphMath(db_path)
    print("K.R.I.S.T.Y. v1.0 Online.")
    print("-" * 50)
    
    user_session_queries = [
        "What is the total price for iPhone 7?",
        "Tell me the price of the MacBook Pro 13",
        "How much is the Apple Watch Series 2 with total price?"
    ]
    
    for query in user_session_queries:
        print("USER  >> {}".format(query))
        response = kristy.handle_query(query)
        print("AGENT >> {}".format(response))
        print("-" * 50)

if __name__ == "__main__":
    main()

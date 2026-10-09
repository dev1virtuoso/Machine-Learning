import sqlite3
import networkx as nx
import sympy
import logging
from nltk.tokenize import word_tokenize

logger = logging.getLogger("KristyEngine")

class KristyGraphMath:
    def __init__(self, db_path="./data/knowledge_base.db"):
        self.kb = nx.DiGraph()
        self.db_path = db_path
        self._load_from_database()

    def _load_from_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name, type, price FROM products")
            for name, p_type, price in cursor.fetchall():
                self.kb.add_node(name, type=p_type, price=price)

            cursor.execute("SELECT value FROM metadata WHERE key='standard_tax_rate'")
            tax_row = cursor.fetchone()
            tax_val = tax_row[0] if tax_row else 0.0
            self.kb.add_node("Tax_Rate", value=tax_val)

            for node, data in self.kb.nodes(data=True):
                if data.get("type") == "Smartphone":
                    self.kb.add_edge(node, "Tax_Rate", relation="subject_to")
            
            conn.close()
        except Exception as e:
            logger.error("Failed to sync Knowledge Graph: %s", e)

    def handle_query(self, text):
        tokens = word_tokenize(text.lower())
        target_entity = next((node for node in self.kb.nodes() if node.lower() in text.lower()), None)
        
        if not target_entity:
            return "I am sorry, I do not have information on that product in my current database."

        entity_data = self.kb.nodes[target_entity]

        if "total" in tokens and "price" in tokens:
            return self._calculate_symbolic_total(target_entity, entity_data)

        return "Product: {} | Type: {} | Base Price: ${:.2f}".format(
            target_entity, entity_data['type'], entity_data['price']
        )

    def _calculate_symbolic_total(self, name, data):
        price = data.get('price')
        if not self.kb.has_edge(name, "Tax_Rate"):
            return "The price for {} is ${:.2f} (No tax records found).".format(name, price)

        tax_rate = self.kb.nodes["Tax_Rate"]['value']
        
        p, t = sympy.symbols('p t')
        formula = p * (1 + t)
        final_price = float(formula.subs({p: price, t: tax_rate}))
        
        return "The total price for {} after {}% tax is ${:.2f}.".format(
            name, int(tax_rate * 100), final_price
        )

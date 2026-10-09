import os
import sys
import json
import sqlite3
from unittest.mock import patch, MagicMock

import pytest
import networkx as nx
import sympy

ROOT = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(ROOT, "core")
sys.path.insert(0, ROOT)
sys.path.insert(0, CORE)

from core.prepare import build_database
from core.kristy import KristyGraphMath
import core.main as main_module

@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "knowledge_base.db")
    yield db_path


@pytest.fixture
def populated_db(temp_db):
    os.makedirs(os.path.dirname(temp_db) or ".", exist_ok=True)
    build_database(db_path=temp_db)
    return temp_db


@pytest.fixture
def engine(populated_db):
    return KristyGraphMath(db_path=populated_db)

class TestBuildDatabase:
    def test_creates_file(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(data_dir / "knowledge_base.db")
        build_database(db_path=db_path)
        assert os.path.exists(db_path)

    def test_tables_exist(self, populated_db):
        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "products" in tables
        assert "metadata" in tables

    def test_product_count(self, populated_db):
        conn = sqlite3.connect(populated_db)
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        conn.close()
        assert count == 5

    def test_product_contents(self, populated_db):
        conn = sqlite3.connect(populated_db)
        rows = conn.execute("SELECT name, type, price FROM products").fetchall()
        conn.close()
        names = {r[0] for r in rows}
        assert "iPhone 7" in names
        assert "MacBook Pro 13" in names
        assert "Apple Watch Series 2" in names
        for _, ptype, price in rows:
            assert price > 0
            assert ptype in {"Smartphone", "Laptop", "Wearable"}

    def test_tax_rate(self, populated_db):
        conn = sqlite3.connect(populated_db)
        val = conn.execute(
            "SELECT value FROM metadata WHERE key='standard_tax_rate'"
        ).fetchone()[0]
        conn.close()
        assert abs(val - 0.08) < 1e-9

    def test_idempotent(self, populated_db):
        build_database(db_path=populated_db)
        conn = sqlite3.connect(populated_db)
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        conn.close()
        assert count == 5

class TestKristyGraphMathInit:
    def test_is_digraph(self, engine):
        assert isinstance(engine.kb, nx.DiGraph)

    def test_nodes_loaded(self, engine):
        nodes = set(engine.kb.nodes())
        assert "iPhone 7" in nodes
        assert "MacBook Pro 13" in nodes
        assert "Tax_Rate" in nodes

    def test_smartphone_has_tax_edge(self, engine):
        for node, data in engine.kb.nodes(data=True):
            if data.get("type") == "Smartphone":
                assert engine.kb.has_edge(node, "Tax_Rate")
            elif node != "Tax_Rate":
                assert not engine.kb.has_edge(node, "Tax_Rate")

    def test_tax_rate_value(self, engine):
        assert abs(engine.kb.nodes["Tax_Rate"]["value"] - 0.08) < 1e-9

    def test_missing_db_no_crash(self, tmp_path, caplog):
        bad = str(tmp_path / "no_such.db")
        with caplog.at_level("ERROR"):
            eng = KristyGraphMath(db_path=bad)
        assert len(eng.kb.nodes) == 0
        assert any("Failed to sync Knowledge Graph" in r.message for r in caplog.records)


class TestHandleQuery:
    def test_unknown_product(self, engine):
        resp = engine.handle_query("What is the price of Nokia 3310?")
        assert "sorry" in resp.lower() or "do not have information" in resp.lower()

    def test_basic_price(self, engine):
        resp = engine.handle_query("Tell me the price of the MacBook Pro 13")
        assert "MacBook Pro 13" in resp
        assert "Laptop" in resp
        assert "1299" in resp

    def test_total_price_with_tax(self, engine):
        resp = engine.handle_query("What is the total price for iPhone 7?")
        assert "iPhone 7" in resp
        assert "700.92" in resp
        assert "tax" in resp.lower()

    def test_total_price_no_tax(self, engine):
        resp = engine.handle_query("What is the total price for MacBook Pro 13?")
        assert "MacBook Pro 13" in resp
        assert "No tax records found" in resp or "1299" in resp

    def test_case_insensitive(self, engine):
        resp = engine.handle_query("what is the total price for iphone 7?")
        assert "700.92" in resp

    def test_empty_query(self, engine):
        resp = engine.handle_query("")
        assert "sorry" in resp.lower() or "do not have information" in resp.lower()


class TestSymbolicCalculation:
    def test_correct_formula(self, engine):
        data = {"price": 100.0, "type": "Smartphone"}
        engine.kb.add_node("TestPhone", **data)
        engine.kb.add_edge("TestPhone", "Tax_Rate")
        resp = engine._calculate_symbolic_total("TestPhone", data)
        assert "108.00" in resp
        assert "8%" in resp

    def test_no_tax_edge_message(self, engine):
        data = {"price": 500.0, "type": "Laptop"}
        engine.kb.add_node("TestLaptop", **data)
        resp = engine._calculate_symbolic_total("TestLaptop", data)
        assert "No tax records found" in resp
        assert "500.00" in resp

    def test_uses_sympy(self, engine):
        data = engine.kb.nodes["iPhone 7"]
        with patch("core.kristy.sympy.symbols") as mock_sym:
            mock_sym.return_value = (sympy.Symbol("p"), sympy.Symbol("t"))
            engine._calculate_symbolic_total("iPhone 7", data)
            mock_sym.assert_called_once_with("p t")
            
class TestMain:
    def test_missing_db(self, capsys):
        with patch("core.main.os.path.exists", return_value=False):
            main_module.main()
        out = capsys.readouterr().out
        assert "CRITICAL" in out
        assert "Database not found" in out

    def test_happy_path(self, capsys):
        with patch("core.main.os.path.exists", return_value=True), \
             patch("core.main.KristyGraphMath") as MockEng:
            inst = MockEng.return_value
            inst.handle_query.side_effect = [
                "resp1", "resp2", "resp3"
            ]
            main_module.main()
            out = capsys.readouterr().out
            assert "K.R.I.S.T.Y. v1.0 Online" in out
            assert "USER  >>" in out
            assert "AGENT >>" in out
            assert inst.handle_query.call_count == 3

@pytest.fixture
def client(populated_db):
    with patch("server.build_database"), \
         patch("server.KristyGraphMath") as MockEng, \
         patch("server.os.path.exists", return_value=True):

        mock_engine = MockEng.return_value
        mock_engine.handle_query.side_effect = lambda t: f"ECHO: {t}"

        if "server" in sys.modules:
            del sys.modules["server"]
        import server
        server.engine = mock_engine

        server.app.config["TESTING"] = True
        with server.app.test_client() as c:
            yield c, mock_engine


class TestServer:
    def test_query_success(self, client):
        c, mock_engine = client
        resp = c.post(
            "/query",
            data=json.dumps({"text": "What is the total price for iPhone 7?"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "response" in data
        assert data["response"].startswith("ECHO:")
        mock_engine.handle_query.assert_called_once()

    def test_empty_text(self, client):
        c, _ = client
        resp = c.post(
            "/query",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "Please provide a query" in resp.get_json()["response"]

    def test_invalid_json(self, client):
        c, _ = client
        resp = c.post(
            "/query",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "Invalid JSON" in resp.get_json()["response"]

    def test_missing_text_key(self, client):
        c, _ = client
        resp = c.post(
            "/query",
            data=json.dumps({"foo": "bar"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

def test_full_pipeline(tmp_path):
    db_path = str(tmp_path / "knowledge_base.db")
    os.makedirs(tmp_path, exist_ok=True)
    build_database(db_path=db_path)

    eng = KristyGraphMath(db_path=db_path)

    r1 = eng.handle_query("What is the total price for iPhone 7?")
    assert "700.92" in r1

    r2 = eng.handle_query("Tell me the price of the MacBook Pro 13")
    assert "1299" in r2
    assert "Laptop" in r2

    r3 = eng.handle_query("How much is a Pixel 9?")
    assert "sorry" in r3.lower() or "do not have information" in r3.lower()


def test_graph_structure(engine):
    assert len(engine.kb.nodes) == 6
    tax_edges = [(u, v) for u, v in engine.kb.edges() if v == "Tax_Rate"]
    assert len(tax_edges) == 3
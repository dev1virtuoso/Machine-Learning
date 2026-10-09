import os
import sqlite3
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call

import pytest
import networkx as nx
import sympy
import sys

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
    data_dir = os.path.dirname(temp_db)
    os.makedirs(data_dir, exist_ok=True)
    build_database(db_path=temp_db)
    return temp_db


@pytest.fixture
def engine(populated_db):
    return KristyGraphMath(db_path=populated_db)

class TestBuildDatabase:
    def test_creates_directory_and_file(self, tmp_path):
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
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 5

    def test_product_contents(self, populated_db):
        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name, type, price FROM products ORDER BY name")
        rows = cursor.fetchall()
        conn.close()

        names = [r[0] for r in rows]
        assert "iPhone 7" in names
        assert "MacBook Pro 13" in names
        assert "Apple Watch Series 2" in names

        for name, p_type, price in rows:
            assert price > 0
            assert p_type in {"Smartphone", "Laptop", "Wearable"}

    def test_tax_rate_metadata(self, populated_db):
        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key='standard_tax_rate'")
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert abs(row[0] - 0.08) < 1e-9

    def test_idempotent_insert(self, populated_db):
        build_database(db_path=populated_db)
        conn = sqlite3.connect(populated_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 5

class TestKristyGraphMathInit:
    def test_graph_is_digraph(self, engine):
        assert isinstance(engine.kb, nx.DiGraph)

    def test_nodes_loaded(self, engine):
        nodes = list(engine.kb.nodes())
        assert "iPhone 7" in nodes
        assert "MacBook Pro 13" in nodes
        assert "Tax_Rate" in nodes

    def test_smartphone_edges_to_tax(self, engine):
        for node, data in engine.kb.nodes(data=True):
            if data.get("type") == "Smartphone":
                assert engine.kb.has_edge(node, "Tax_Rate")
            elif node != "Tax_Rate":
                assert not engine.kb.has_edge(node, "Tax_Rate")

    def test_tax_rate_value(self, engine):
        assert "Tax_Rate" in engine.kb.nodes
        assert abs(engine.kb.nodes["Tax_Rate"]["value"] - 0.08) < 1e-9

    def test_missing_db_does_not_crash(self, tmp_path, caplog):
        bad_path = str(tmp_path / "nonexistent.db")
        with caplog.at_level("ERROR"):
            eng = KristyGraphMath(db_path=bad_path)
        assert len(eng.kb.nodes) == 0
        assert any("Failed to sync Knowledge Graph" in rec.message for rec in caplog.records)


class TestHandleQuery:
    def test_unknown_product(self, engine):
        resp = engine.handle_query("What is the price of the Nokia 3310?")
        assert "sorry" in resp.lower() or "do not have information" in resp.lower()

    def test_basic_price_query(self, engine):
        resp = engine.handle_query("Tell me the price of the MacBook Pro 13")
        assert "MacBook Pro 13" in resp
        assert "Laptop" in resp
        assert "1299.00" in resp or "1299" in resp

    def test_total_price_smartphone_with_tax(self, engine):
        resp = engine.handle_query("What is the total price for iPhone 7?")
        assert "iPhone 7" in resp
        assert "tax" in resp.lower()
        assert "700.92" in resp

    def test_total_price_laptop_no_tax(self, engine):
        resp = engine.handle_query("What is the total price for MacBook Pro 13?")
        assert "MacBook Pro 13" in resp
        assert "No tax records found" in resp or "1299.00" in resp

    def test_case_insensitive_matching(self, engine):
        resp = engine.handle_query("what is the total price for iphone 7?")
        assert "iPhone 7" in resp
        assert "700.92" in resp

    def test_partial_name_in_sentence(self, engine):
        resp = engine.handle_query("How much is the Apple Watch Series 2 with total price?")
        assert "Apple Watch Series 2" in resp

    def test_empty_query(self, engine):
        resp = engine.handle_query("")
        assert "sorry" in resp.lower() or "do not have information" in resp.lower()


class TestSymbolicCalculation:
    def test_sympy_formula(self, engine):
        data = {"price": 100.0, "type": "Smartphone"}
        engine.kb.add_node("TestPhone", **data)
        engine.kb.add_edge("TestPhone", "Tax_Rate", relation="subject_to")

        resp = engine._calculate_symbolic_total("TestPhone", data)
        assert "108.00" in resp
        assert "8%" in resp

    def test_no_tax_edge(self, engine):
        data = {"price": 500.0, "type": "Laptop"}
        engine.kb.add_node("TestLaptop", **data)
        resp = engine._calculate_symbolic_total("TestLaptop", data)
        assert "No tax records found" in resp
        assert "500.00" in resp

    def test_sympy_symbols_used(self, engine):
        data = engine.kb.nodes["iPhone 7"]
        with patch("kristy.sympy.symbols") as mock_symbols:
            mock_symbols.return_value = (sympy.Symbol("p"), sympy.Symbol("t"))
            engine._calculate_symbolic_total("iPhone 7", data)
            mock_symbols.assert_called_once_with("p t")

class TestMain:
    def test_main_missing_db(self, tmp_path, capsys):
        with patch("main.os.path.exists", return_value=False):
            main_module.main()
        captured = capsys.readouterr()
        assert "CRITICAL" in captured.out
        assert "Database not found" in captured.out

    def test_main_happy_path(self, populated_db, capsys):
        with patch("core.main.os.path.exists", return_value=True), \
            patch("core.main.KristyGraphMath") as MockEngine:

            instance = MockEngine.return_value
            instance.handle_query.side_effect = [
                "The total price for iPhone 7 after 8% tax is $700.92.",
                "Product: MacBook Pro 13 | Type: Laptop | Base Price: $1299.00",
                "The price for Apple Watch Series 2 is $369.00 (No tax records found)."
            ]

            main_module.main()

            captured = capsys.readouterr()
            assert "K.R.I.S.T.Y. v1.0 Online" in captured.out
            assert "USER  >>" in captured.out
            assert "AGENT >>" in captured.out
            assert MockEngine.called
            assert instance.handle_query.call_count == 3

class TestKristyGUIHelpers:

    def test_append_text_logic(self):
        mock_chat = MagicMock()
        mock_chat.configure = MagicMock()
        mock_chat.insert = MagicMock()
        mock_chat.see = MagicMock()

        def append_text(text, user=True):
            mock_chat.configure(state="normal")
            prefix = "You: " if user else "K.R.I.S.T.Y.: "
            mock_chat.insert("end", f"{prefix}{text}\n")
            mock_chat.configure(state="disabled")
            mock_chat.see("end")

        append_text("Hello", user=True)
        mock_chat.insert.assert_called_with("end", "You: Hello\n")

        append_text("Hi there", user=False)
        mock_chat.insert.assert_called_with("end", "K.R.I.S.T.Y.: Hi there\n")

def test_full_pipeline(tmp_path):
    db_path = str(tmp_path / "knowledge_base.db")
    build_database(db_path=db_path)

    engine = KristyGraphMath(db_path=db_path)

    resp1 = engine.handle_query("What is the total price for iPhone 7?")
    assert "700.92" in resp1

    resp2 = engine.handle_query("Tell me the price of the MacBook Pro 13")
    assert "1299.00" in resp2
    assert "Laptop" in resp2

    resp3 = engine.handle_query("How much is a Pixel 9?")
    assert "sorry" in resp3.lower() or "do not have information" in resp3.lower()


def test_graph_structure_integrity(engine):
    assert len(engine.kb.nodes) == 6

    tax_edges = [
        (u, v) for u, v, d in engine.kb.edges(data=True)
        if v == "Tax_Rate"
    ]
    assert len(tax_edges) == 3
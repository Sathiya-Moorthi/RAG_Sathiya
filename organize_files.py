import os
import shutil
from pathlib import Path

def move_files():
    root = Path(".")
    
    # Define mappings (Source -> Destination)
    moves = {
        # Core Modules
        "rag_config.py": "core/config.py",
        "ingest_all_doc.py": "core/ingestion.py",
        "retrieve_with_web_fallback.py": "core/retriever.py",
        "kg_retrieval.py": "core/knowledge_graph.py",
        "compare_rag_kg.py": "core/comparator.py",
        
        # Scripts
        "ingest_to_neo4j.py": "scripts/ingest_to_neo4j.py",
        "check_neo4j.py": "scripts/check_neo4j.py",
        "build_kg.py": "scripts/build_kg.py",
        "drop_and_recreate_table.py": "scripts/drop_and_recreate_table.py",
        "fix_table_dimensions.py": "scripts/fix_table_dimensions.py",
        "test_db_connection.py": "scripts/test_db_connection.py",
        "rag_app.py": "scripts/legacy_rag_app.py"
    }
    
    # Create directories if not exist (double check)
    os.makedirs("core", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    
    # Create __init__.py in core
    (root / "core" / "__init__.py").touch()

    for src, dst in moves.items():
        src_path = root / src
        dst_path = root / dst
        
        if src_path.exists():
            print(f"Moving {src} -> {dst}")
            shutil.move(str(src_path), str(dst_path))
        else:
            print(f"Warning: {src} not found.")

if __name__ == "__main__":
    move_files()

"""
Capability Store: Version control and intelligent reuse of generated code.
Stores capabilities in Git with semantic search via ChromaDB.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import git

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from core.data_models import GeneratedCapability, ExecutionResult
import config

logger = logging.getLogger(__name__)


class CapabilityStore:
    """Store, version control, and retrieve generated capabilities."""

    def __init__(self, store_path: str = None):
        """
        Initialize capability store.

        Args:
            store_path: Path to capability store directory
        """
        self.store_path = Path(store_path) if store_path else config.CAPABILITY_STORE_PATH
        self.capabilities_dir = self.store_path / "capabilities"
        self.db_path = self.store_path / "capability_db"

        # Create directory structure
        self.capabilities_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Git repository
        self.repo = self._init_git_repo()

        # Initialize vector database for semantic search
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )

        # Load or create vector store
        self.vectorstore = self._init_vectorstore()

        logger.info(f"Capability store initialized at {self.store_path}")

    def _init_git_repo(self) -> git.Repo:
        """Initialize or open Git repository."""
        try:
            repo = git.Repo(self.store_path)
            logger.debug("Opened existing Git repository")
            return repo
        except git.exc.InvalidGitRepositoryError:
            repo = git.Repo.init(self.store_path)
            logger.info("Initialized new Git repository")

            # Create initial commit
            gitignore_path = self.store_path / ".gitignore"
            gitignore_path.write_text("*.pyc\n__pycache__/\n.DS_Store\n")
            repo.index.add([".gitignore"])
            repo.index.commit("Initial commit: Setup capability store")

            return repo

    def _init_vectorstore(self) -> Chroma:
        """Initialize or load ChromaDB vector store."""
        try:
            vectorstore = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=self.embeddings,
                collection_name="capabilities"
            )
            logger.debug("Loaded existing capability vector database")
            return vectorstore
        except Exception as e:
            logger.debug(f"Creating new capability vector database: {e}")
            vectorstore = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=self.embeddings,
                collection_name="capabilities"
            )
            return vectorstore

    def save_capability(
        self,
        request: str,
        capability: GeneratedCapability,
        execution_result: ExecutionResult
    ) -> str:
        """
        Save capability with version control.

        Args:
            request: User's original request
            capability: Generated capability object
            execution_result: Result of execution

        Returns:
            Capability ID
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_hash = hashlib.md5(request.encode()).hexdigest()[:6]
        cap_id = f"cap_{timestamp}_{request_hash}"

        logger.info(f"Saving capability {cap_id} for request: '{request[:50]}'")

        # Save Python code
        code_path = self.capabilities_dir / f"{cap_id}.py"
        code_path.write_text(capability.code)

        # Save metadata
        metadata = {
            "request": request,
            "created_at": datetime.now().isoformat(),
            "imports": capability.imports_used,
            "success": execution_result.success,
            "execution_time": execution_result.execution_time,
            "reuse_count": 0,
            "last_used": None,
            "description": capability.description
        }

        metadata_path = self.capabilities_dir / f"{cap_id}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Git commit
        try:
            self.repo.index.add([str(code_path.relative_to(self.store_path)),
                                  str(metadata_path.relative_to(self.store_path))])
            commit_msg = f"Add capability: {request[:50]}\n\nID: {cap_id}"
            commit = self.repo.index.commit(commit_msg)
            logger.info(f"Git commit: {commit.hexsha[:7]}")
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")

        # Index in vector database
        try:
            self.vectorstore.add_texts(
                texts=[request],
                metadatas=[{
                    "cap_id": cap_id,
                    "success": execution_result.success,
                    "execution_time": execution_result.execution_time,
                    "created_at": metadata["created_at"]
                }],
                ids=[cap_id]
            )
            logger.info(f"Indexed in vector database")
        except Exception as e:
            logger.warning(f"Vector indexing failed: {e}")

        return cap_id

    def search_similar(
        self,
        request: str,
        threshold: float = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Search for similar capabilities.

        Args:
            request: User's request
            threshold: Minimum similarity score (default from config)
            top_k: Number of results to return

        Returns:
            List of similar capabilities with metadata
        """
        threshold = threshold or config.CAPABILITY_SIMILARITY_THRESHOLD

        logger.info(f"Searching for similar capabilities to: '{request[:50]}'")

        try:
            # Search with scores
            results = self.vectorstore.similarity_search_with_score(
                request,
                k=top_k
            )

            similar_capabilities = []

            for doc, distance in results:
                # Convert distance to similarity (ChromaDB uses L2 distance)
                # For normalized vectors, similarity = 1 - (distance^2 / 2)
                similarity = 1.0 - (distance / 2.0)

                # Filter by threshold
                if similarity >= threshold:
                    cap_id = doc.metadata.get("cap_id")

                    # Only include successful capabilities
                    if doc.metadata.get("success", False):
                        similar_capabilities.append({
                            "cap_id": cap_id,
                            "similarity": similarity,
                            "metadata": doc.metadata,
                            "request": doc.page_content
                        })

            logger.info(f"Found {len(similar_capabilities)} capabilities above threshold {threshold}")

            if similar_capabilities:
                best = similar_capabilities[0]
                logger.debug(f"Top match: {best['cap_id']} (similarity: {best['similarity']:.3f})")

            return similar_capabilities

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def load_capability(self, cap_id: str) -> GeneratedCapability:
        """
        Load capability by ID.

        Args:
            cap_id: Capability ID

        Returns:
            GeneratedCapability object

        Raises:
            ValueError: If capability not found
        """
        code_path = self.capabilities_dir / f"{cap_id}.py"
        metadata_path = self.capabilities_dir / f"{cap_id}.json"

        if not code_path.exists() or not metadata_path.exists():
            raise ValueError(f"Capability {cap_id} not found")

        logger.info(f"Loading capability {cap_id}")

        # Load code
        code = code_path.read_text()

        # Load metadata
        metadata = json.loads(metadata_path.read_text())

        return GeneratedCapability(
            code=code,
            description=metadata.get("description", ""),
            imports_used=metadata.get("imports", []),
            estimated_runtime=metadata.get("estimated_runtime", "unknown")
        )

    def increment_reuse(self, cap_id: str):
        """
        Increment reuse count for a capability.

        Args:
            cap_id: Capability ID
        """
        metadata_path = self.capabilities_dir / f"{cap_id}.json"

        if not metadata_path.exists():
            logger.warning(f"Cannot increment reuse: {cap_id} not found")
            return

        # Load metadata
        metadata = json.loads(metadata_path.read_text())

        # Update stats
        metadata["reuse_count"] = metadata.get("reuse_count", 0) + 1
        metadata["last_used"] = datetime.now().isoformat()

        # Save updated metadata
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Git commit
        try:
            self.repo.index.add([str(metadata_path.relative_to(self.store_path))])
            commit_msg = f"Reuse capability {cap_id} (count: {metadata['reuse_count']})"
            self.repo.index.commit(commit_msg)
        except Exception as e:
            logger.warning(f"Git commit for reuse failed: {e}")

        logger.info(f"Capability {cap_id} reused (total: {metadata['reuse_count']})")

    def get_capability_stats(self, cap_id: str) -> Dict:
        """Get usage statistics for a capability."""
        metadata_path = self.capabilities_dir / f"{cap_id}.json"

        if not metadata_path.exists():
            raise ValueError(f"Capability {cap_id} not found")

        return json.loads(metadata_path.read_text())

    def list_all_capabilities(self, sort_by: str = "created_at") -> List[Dict]:
        """
        List all stored capabilities.

        Args:
            sort_by: Sort field ('created_at', 'reuse_count', 'last_used')

        Returns:
            List of capability metadata dicts
        """
        capabilities = []

        for metadata_file in self.capabilities_dir.glob("*.json"):
            try:
                metadata = json.loads(metadata_file.read_text())
                metadata["cap_id"] = metadata_file.stem
                capabilities.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load {metadata_file}: {e}")

        # Sort
        if sort_by == "reuse_count":
            capabilities.sort(key=lambda x: x.get("reuse_count", 0), reverse=True)
        elif sort_by == "last_used":
            capabilities.sort(key=lambda x: x.get("last_used") or "", reverse=True)
        else:  # created_at
            capabilities.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return capabilities

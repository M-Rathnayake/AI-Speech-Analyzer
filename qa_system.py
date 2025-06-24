from haystack import Pipeline, Document
from haystack.nodes import EmbeddingRetriever, FARMReader
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self):
        # Updated EmbeddingRetriever initialization
        self.retriever = EmbeddingRetriever(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            model_format="sentence_transformers",
            progress_bar=False
        )
        
        self.reader = FARMReader(
            model_name_or_path="deepset/roberta-base-squad2",
            use_gpu=False
        )
        
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Build Haystack QA pipeline."""
        pipeline = Pipeline()
        pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        pipeline.add_node(component=self.reader, name="Reader", inputs=["Retriever"])
        return pipeline

    def answer(self, text: str, question: str) -> Optional[str]:
        """Answer question about given text."""
        try:
            doc = Document(content=text)
            result = self.pipeline.run(query=question, documents=[doc])
            return result["answers"][0].answer if result["answers"] else None
        except Exception as e:
            logger.error(f"QA failed: {e}")
            return None
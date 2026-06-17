import polars as pl
from typing  import Protocol, Any

class FootballModel(Protocol):
    def predictScores(
        self,
        df: pl.DataFrame, 
        teamANameExpr: pl.Expr, 
        teamBNameExpr: pl.Expr,
        teamAScoreExpr: pl.Expr,
        teamBScoreExpr: pl.Expr,
        **kwargs: Any,

    ) -> pl.Series:
        ...
import tensorflow as tf
import polars as pl
import numpy as np

from typing import List
from scipy.stats import poisson
from math import factorial

def poissonPdf(k: int, param: float):
    return np.exp(-param) * (param**k) / factorial(k)

class PoissonModel(tf.keras.Model):
    def __init__(
        self,
        teamList: List[str]
    ):
        super().__init__()
        self.teamList = teamList
        self.numTeams = len(teamList)
        self.teamParams = self.add_weight(
            name="team_params",
            shape=(2 * self.numTeams, 1),
            initializer=tf.random_normal_initializer(),
            trainable=True
        )
        self.awayHomeMultiplier = self.add_weight(
            name="away_home_multiplier",
            shape=(1, 2),
            initializer=tf.random_normal_initializer(),
            trainable=True
        )

    def call(self, input):
        attackingParams = self.teamParams[:self.numTeams]
        defendingParams = self.teamParams[self.numTeams:2*self.numTeams]

        attackingOneHotEncoder = input[:,:self.numTeams]
        defendingOneHotEncoder = input[:,self.numTeams:2*self.numTeams]
        score = input[:,-1]
        awayHomeMultiplier = tf.reduce_sum(input[:, -3:-1] * self.awayHomeMultiplier, axis=1, keepdims=True)
        scoreExpanded = tf.expand_dims(score, axis=1)

        params = tf.exp(awayHomeMultiplier + tf.matmul(attackingOneHotEncoder, attackingParams) - tf.matmul(defendingOneHotEncoder, defendingParams))
        
        poissonLogLikelyHood = -params + tf.math.log(params) * scoreExpanded - tf.math.lgamma(scoreExpanded + 1)

        return tf.reduce_sum(poissonLogLikelyHood)

    def fitPoisson(
        self,
        data: pl.DataFrame,
        scopeExpr: pl.Expr,
        homeTeamExpr: pl.Expr,
        awayTeamExpr: pl.Expr,
        homeScoreExpr: pl.Expr,
        awayScoreExpr: pl.Expr,
        matchStatusExpr: pl.Expr,
        numEpochs: int = 1000,
    ):
        dataInScope = data.filter(scopeExpr)

        homeTeamColumnName = "home_team"
        awayTeamColumnName = "away_team"
        homeScoreColumnName = "home_score"
        awayScoreColumnName = "away_score"
        statusColumnName = "status"

        dataInScope = dataInScope.with_columns(
            homeTeamExpr.alias(homeTeamColumnName),
            awayTeamExpr.alias(awayTeamColumnName),
            homeScoreExpr.alias(homeScoreColumnName),
            awayScoreExpr.alias(awayScoreColumnName),
            matchStatusExpr.alias(statusColumnName)
        )

        teamNumerotation = {team: i for i, team in enumerate(self.teamList)}
        matchMatrice = np.zeros((dataInScope.height*2, self.numTeams*2 + 3))
        for i, row in enumerate(dataInScope.iter_rows(named=True)):
            #home team score
            matchMatrice[2*i, teamNumerotation[str(row[homeTeamColumnName])]] = 1
            matchMatrice[2*i, self.numTeams+teamNumerotation[str(row[awayTeamColumnName])]] = 1
            matchMatrice[2*i, -1] = row[homeScoreColumnName]
            #away team score
            matchMatrice[2*i + 1, self.numTeams+teamNumerotation[str(row[homeTeamColumnName])]] = 1
            matchMatrice[2*i + 1, teamNumerotation[str(row[awayTeamColumnName])]] = 1
            matchMatrice[2*i + 1, -1] = row[awayScoreColumnName]

            if not row[statusColumnName]:
                matchMatrice[2*i, -3] = 1
                matchMatrice[2*i + 1, -2] = 1

        optimizer = tf.keras.optimizers.Adam()
        X = tf.constant(matchMatrice, dtype=tf.float32)
        for i in range(numEpochs):
            print(f"Epoch {i+1}/{numEpochs}", end="\r")
            with tf.GradientTape() as tape:
                pred = -self(X)
                loss = pred
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def getTeamParamsDf(self) -> pl.DataFrame:
        attackingParams = [np.exp(self.teamParams.numpy())[i][0] for i in range(self.numTeams)]
        defendingParams = [np.exp(self.teamParams.numpy())[self.numTeams+i][0] for i in range(self.numTeams)]
        res = pl.DataFrame({
            "team": self.teamList,
            "attackingParams": attackingParams, 
            "defendingParams": defendingParams
        }
        )
        return res

    def getHomeMultiplier(self) -> float:
        return np.exp(self.awayHomeMultiplier[0][0])

    def getAwayMultiplier(self) -> float:
        return np.exp(self.awayHomeMultiplier[0][1])

    def predictScore(
        self,
        df: pl.DataFrame, 
        teamANameExpr: pl.Expr, 
        teamBNameExpr: pl.Expr,
        teamAScoreExpr: pl.Expr,
        teamBScoreExpr: pl.Expr,
        homeTeamExpr: pl.Expr
    ):
        teamParamsDfTeamA = self.getTeamParamsDf().rename(
            {
                "attackingParams" : "attackingParams_teamA",
                "defendingParams" : "defendingParams_teamA"
            }
        )
        teamParamsDfTeamB = self.getTeamParamsDf().rename(
            {
                "attackingParams" : "attackingParams_teamB",
                "defendingParams" : "defendingParams_teamB"
            }
        )
        dfJoined = df.join(
            teamParamsDfTeamA, 
            left_on=teamANameExpr, 
            right_on=pl.col("team")
        ) 
        dfJoined = dfJoined.join(
            teamParamsDfTeamB, 
            left_on=teamBNameExpr, 
            right_on=pl.col("team")
        ) 

        dfJoined = dfJoined.with_columns(
            teamAMultiplier = pl.when(homeTeamExpr==teamANameExpr)
            .then(self.getHomeMultiplier())
            .when(homeTeamExpr==teamBNameExpr)
            .then(self.getAwayMultiplier())
            .otherwise(pl.lit(1.0)),
            teamBMultiplier = pl.when(homeTeamExpr==teamBNameExpr)
            .then(self.getHomeMultiplier())
            .when(homeTeamExpr==teamANameExpr)
            .then(self.getAwayMultiplier())
            .otherwise(pl.lit(1.0)),
        )

        dfJoined = dfJoined.with_columns(
            poissonParamTeamA = pl.col("teamAMultiplier")*pl.col("attackingParams_teamA")/pl.col("defendingParams_teamB"),
            poissonParamTeamB = pl.col("teamBMultiplier")*pl.col("attackingParams_teamB")/pl.col("defendingParams_teamA")
        )

        dfJoined = dfJoined.with_columns(
            likelyhoodScoreTeamA = 
            pl.struct([teamAScoreExpr.alias("teamAScore"), "poissonParamTeamA"])
            .map_elements(lambda row: poissonPdf(row["teamAScore"], row["poissonParamTeamA"])),
            likelyhoodScoreTeamB = 
            pl.struct([teamBScoreExpr.alias("teamBScore"), "poissonParamTeamB"])
            .map_elements(lambda row: poissonPdf(row["teamBScore"], row["poissonParamTeamB"]))
        )

        dfJoined = dfJoined.with_columns(
            likelyhoodScore = pl.col("likelyhoodScoreTeamA")*pl.col("likelyhoodScoreTeamB")
        )

        return dfJoined["likelyhoodScore"]
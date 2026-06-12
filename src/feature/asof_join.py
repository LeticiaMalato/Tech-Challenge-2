import pandas as pd


def asof_join_properties(
    interactions: pd.DataFrame,
    properties: pd.DataFrame,
) -> pd.DataFrame:
 #Anexa as propriedades dos itens às interações usando um merge_asof. A função `asof_join_properties` realiza uma junção temporal entre o DataFrame de interações e o DataFrame de propriedades dos itens, associando a cada interação as características do item mais recentes disponíveis até aquele momento. O processo envolve pivotar o DataFrame de propriedades para um formato wide, ordenar ambos os DataFrames por timestamp, e usar `pd.merge_asof` para fazer a junção, garantindo que cada interação receba as informações do item correspondentes ao seu timestamp.
    # Pivot: transforma o formato long em wide (uma coluna por propriedade)
    props_wide = (
        properties
        .pivot_table(
            index=["itemid", "timestamp"],
            columns="property",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    props_wide.columns.name = None

    # Ordena ambos por timestamp — requisito do merge_asof
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    props_wide = props_wide.sort_values("timestamp").reset_index(drop=True)

    # merge_asof: para cada linha de interactions, encontra a linha mais
    # recente de props_wide com timestamp <= timestamp do evento
    merged = pd.merge_asof(
        interactions,
        props_wide,
        on="timestamp",
        by="itemid",
        direction="backward",
    )

    return merged
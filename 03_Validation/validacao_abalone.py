#!/usr/bin/env python3
"""Treina, valida e prepara a submissao da atividade

O script usa somente o arquivo de treino para escolher e avaliar o modelo. A
base abalone_app.csv e lida apenas depois do ajuste final, para evitar
vazamento de dados da avaliacao externa.

Uso normal:
    python validacao_abalone.py

Para enviar, depois de revisar os resultados:
    python validacao_abalone.py --enviar --chave puxafrango
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / ".ml_packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

try:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVC
except ModuleNotFoundError as error:
    package = error.name.split(".")[0]
    raise SystemExit(
        f"Dependencia ausente: {package}. Instale as dependencias com:\n"
        "  python -m pip install -r requirements.txt"
    ) from error


TARGET = "type"
RAW_FEATURES = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
SUBMISSION_URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEFAULT_KEY = os.environ.get("MLCLASS_DEV_KEY", "puxafrango")
EPSILON = 1e-9


def check_columns(data: pd.DataFrame, expected: list[str], source: Path) -> None:
    """Falha cedo quando o CSV nao tem o esquema esperado."""
    missing = sorted(set(expected) - set(data.columns))
    unexpected = sorted(set(data.columns) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"faltando: {', '.join(missing)}")
        if unexpected:
            details.append(f"inesperadas: {', '.join(unexpected)}")
        raise ValueError(f"Colunas invalidas em {source.name} ({'; '.join(details)}).")


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta atributos fisicos sem alterar ou usar a classe alvo.

    Volume e proporcoes de peso ajudam a separar animais de tamanho parecido,
    mas em diferentes estagios de maturidade. A constante pequena impede
    divisao por zero se uma futura base tiver uma medida igual a zero.
    """
    features = data.loc[:, RAW_FEATURES].copy()

    length = features["length"]
    diameter = features["diameter"]
    height = features["height"]
    whole = features["whole_weight"]
    shucked = features["shucked_weight"]
    viscera = features["viscera_weight"]
    shell = features["shell_weight"]

    features["volume"] = length * diameter * height
    features["density"] = whole / (features["volume"] + EPSILON)
    features["meat_weight"] = shucked + viscera
    features["meat_fraction"] = (shucked + viscera) / (whole + EPSILON)
    features["shell_fraction"] = shell / (whole + EPSILON)
    features["shell_to_meat"] = shell / (shucked + viscera + EPSILON)
    features["height_to_length"] = height / (length + EPSILON)
    return features


def build_model(features: pd.DataFrame) -> Pipeline:
    """Monta o pipeline vencedor da validacao cruzada estratificada.

    A SVM RBF (C=2.5; gamma=0.04) foi selecionada comparando-a com regressao
    logistica, k-NN, Random Forest, Extra Trees e Gradient Boosting. Escala e
    one-hot encoding ficam dentro do pipeline para que cada dobra de validacao
    calcule seus parametros somente com o proprio conjunto de treino.
    """
    numeric = [column for column in features.columns if column != "sex"]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("sex", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
        ],
        remainder="drop",
    )
    classifier = SVC(
        kernel="rbf",
        C=2.5,
        gamma=0.04,
        decision_function_shape="ovr",
        cache_size=512,
    )
    return Pipeline(steps=[("preprocess", preprocessing), ("classifier", classifier)])


def validate(model: Pipeline, features: pd.DataFrame, target: pd.Series, folds: int, seed: int) -> dict:
    """Produz previsoes out-of-fold e metricas honestas de generalizacao."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out_of_fold = cross_val_predict(model, features, target, cv=splitter, n_jobs=1)
    labels = sorted(target.unique())
    matrix = confusion_matrix(target, out_of_fold, labels=labels)
    report = classification_report(
        target,
        out_of_fold,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target, out_of_fold)),
        "balanced_accuracy": float(balanced_accuracy_score(target, out_of_fold)),
        "f1_macro": float(f1_score(target, out_of_fold, average="macro")),
        "labels": labels,
        "confusion_matrix": matrix,
        "report": report,
        "predictions": out_of_fold,
    }


def write_validation_artifacts(result: dict, target: pd.Series, folds: int, seed: int) -> None:
    """Salva relatorio legivel e tabelas que podem entrar na entrega."""
    labels = result["labels"]
    pd.DataFrame(
        result["confusion_matrix"],
        index=[f"real_{label}" for label in labels],
        columns=[f"previsto_{label}" for label in labels],
    ).to_csv(ROOT / "matriz_confusao_validacao.csv", index_label="classe")

    report = pd.DataFrame(result["report"]).transpose()
    report.to_csv(ROOT / "relatorio_por_classe_validacao.csv", index_label="classe")

    lines = [
        "RELATORIO DE VALIDACAO - ATIVIDADE 03",
        "",
        "Modelo: SVM com kernel RBF (C=2.5; gamma=0.04).",
        "Pre-processamento: mediana + padronizacao dos numericos e one-hot para sex.",
        "Atributos derivados: volume, densidade, massa de carne e proporcoes de peso.",
        f"Metodo: validacao cruzada estratificada com {folds} dobras (semente {seed}).",
        f"Acuracia out-of-fold: {result['accuracy']:.4%}",
        f"Acuracia balanceada out-of-fold: {result['balanced_accuracy']:.4%}",
        f"F1 macro out-of-fold: {result['f1_macro']:.4%}",
        "",
        "Distribuicao real das classes:",
        target.value_counts().sort_index().to_string(),
        "",
        "Matriz de confusao (linhas = real; colunas = previsto):",
        pd.DataFrame(
            result["confusion_matrix"],
            index=[f"real_{label}" for label in labels],
            columns=[f"previsto_{label}" for label in labels],
        ).to_string(),
        "",
        "As metricas sao de previsoes feitas em dobras que nao participaram do treino.",
        "O resultado do servidor pode variar, pois ele avalia uma base externa separada.",
    ]
    (ROOT / "relatorio_validacao.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions(predictions) -> str:
    """Gera os artefatos de previsao, inclusive o JSON aceito pelo servidor."""
    values = [int(value) for value in predictions]
    pd.DataFrame({TARGET: values}).to_csv(ROOT / "previsoes_abalone.csv", index=False)
    payload = json.dumps(values, separators=(",", ":"))
    (ROOT / "previsoes_abalone.json").write_text(payload + "\n", encoding="utf-8")
    return payload


def send_predictions(dev_key: str, prediction_json: str) -> str:
    """Envia uma unica requisicao no formato usado nas atividades anteriores."""
    form_data = urlencode({"dev_key": dev_key, "predictions": prediction_json}).encode("utf-8")
    request = Request(SUBMISSION_URL, data=form_data, method="POST")
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validacao e submissao do classificador de abalones.")
    parser.add_argument("--treino", type=Path, default=ROOT / "abalone_dataset.csv")
    parser.add_argument("--app", type=Path, default=ROOT / "abalone_app.csv")
    parser.add_argument("--dobras", type=int, default=5, help="Numero de dobras estratificadas (padrao: 5).")
    parser.add_argument("--semente", type=int, default=20260930)
    parser.add_argument("--sem-validacao", action="store_true", help="Pula a validacao e apenas gera previsoes.")
    parser.add_argument("--enviar", action="store_true", help="Envia as previsoes ao servidor do curso.")
    parser.add_argument("--chave", default=DEFAULT_KEY, help="Chave da equipe (padrao: MLCLASS_DEV_KEY ou puxafrango).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dobras < 2:
        raise ValueError("--dobras deve ser pelo menos 2.")

    train_path = args.treino.resolve()
    app_path = args.app.resolve()
    train = pd.read_csv(train_path)
    app = pd.read_csv(app_path)
    check_columns(train, RAW_FEATURES + [TARGET], train_path)
    check_columns(app, RAW_FEATURES, app_path)

    features = make_features(train)
    target = train[TARGET].astype(int)
    if target.nunique() != 3 or not target.isin([1, 2, 3]).all():
        raise ValueError("A coluna type do treino deve conter exatamente as classes 1, 2 e 3.")
    if target.value_counts().min() < args.dobras:
        raise ValueError("Ha poucos exemplos em alguma classe para o numero de dobras solicitado.")

    model = build_model(features)
    print(f"Treino: {len(train)} exemplos | classes: {target.value_counts().sort_index().to_dict()}")
    if not args.sem_validacao:
        result = validate(model, features, target, args.dobras, args.semente)
        write_validation_artifacts(result, target, args.dobras, args.semente)
        print("\nValidacao cruzada estratificada (previsoes out-of-fold):")
        print(f"  Acuracia:            {result['accuracy']:.2%}")
        print(f"  Acuracia balanceada: {result['balanced_accuracy']:.2%}")
        print(f"  F1 macro:            {result['f1_macro']:.2%}")
        print("  Arquivos: relatorio_validacao.txt, matriz_confusao_validacao.csv e relatorio_por_classe_validacao.csv")

    model.fit(features, target)
    app_features = make_features(app)
    predictions = model.predict(app_features)
    prediction_json = write_predictions(predictions)
    print(f"\nPrevisoes geradas: {len(predictions)} linhas em previsoes_abalone.csv e previsoes_abalone.json")
    print(f"Distribuicao prevista: {pd.Series(predictions).value_counts().sort_index().to_dict()}")

    if args.enviar:
        print(f"\nEnviando para {SUBMISSION_URL} com a chave '{args.chave}'...")
        try:
            reply = send_predictions(args.chave, prediction_json)
        except Exception as error:  # A resposta do servidor e a parte relevante para o usuario.
            raise SystemExit(f"Falha no envio: {error}") from error
        (ROOT / "resposta_servidor.txt").write_text(reply + "\n", encoding="utf-8")
        print("Resposta salva em resposta_servidor.txt:\n" + reply)
    else:
        print("\nNada foi enviado. Revise os artefatos e use --enviar somente quando a equipe decidir submeter.")


if __name__ == "__main__":
    main()

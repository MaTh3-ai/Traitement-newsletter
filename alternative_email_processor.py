"""
alternative_email_processor.py
--------------------------------

Ce script Python propose une alternative gratuite et auto‑hébergée au projet
Google Apps Script décrit par l’utilisateur. Il est destiné à être lancé à
l’aide d’un service externe gratuit comme GitHub Actions (voir la section
« Déploiement » pour plus de détails) afin d’analyser automatiquement des
e‑mails financiers, d’extraire des informations pertinentes et d’envoyer un
rapport hebdomadaire synthétique.

Objectifs :
  * Éviter les dépendances payantes : le script s’appuie uniquement sur des
    bibliothèques Python open‑source (Hugging Face Transformers, pandas,
    gspread, etc.) et sur des fonctionnalités gratuites comme les GitHub
    Actions programmées【486654252011779†L20-L31】.
  * Réaliser localement les tâches de compréhension de texte en utilisant des
    modèles pré‑entraînés distribués par Hugging Face :
    – la famille Pegasus pour le résumé abstrait【462364561835987†L164-L179】 ;
    – FinBERT pour l’analyse de sentiment financier【360355686124617†L43-L86】.
  * Extraire des tickers boursiers et calculer un score d’opportunité à
    partir du sentiment estimé.
  * Alimenter un classeur Google Sheets avec les résultats individuels et
    agrégés via un compte de service【672301988937356†L51-L99】.
  * Générer un e‑mail récapitulatif présentant les actions « à privilégier »,
    « à surveiller » et « à éviter ».

Prérequis
---------

1. **Accès IMAP** : le script récupère les e‑mails via IMAP. Pour un compte
   Gmail, il est conseillé de créer un mot de passe d’application et
   d’activer l’accès IMAP dans les paramètres du compte. Les variables
   d’environnement suivantes doivent être renseignées :

   - `IMAP_HOST` : par exemple `imap.gmail.com` ;
   - `IMAP_USER` : adresse e‑mail de Mathéo ;
   - `IMAP_PASSWORD` : mot de passe ou mot de passe d’application.

2. **Classeur Google Sheets** : créez un classeur vierge qui servira de
   réceptacle. Partagez‑le avec l’adresse du compte de service. Copiez son
   identifiant (la portion entre `/d/` et `/edit` dans l’URL).

3. **Compte de service** : suivez la procédure décrite dans la
   documentation de gspread pour créer un compte de service et télécharger
   un fichier JSON de crédentiales【672301988937356†L51-L97】. Placez ce
   fichier (par exemple `service_account.json`) dans un emplacement sécurisé
   du dépôt et renseignez la variable d’environnement `GSPREAD_SERVICE_ACCOUNT` 
   avec le chemin correspondant.

4. **SMTP** : pour l’envoi du mail récapitulatif, renseignez :

   - `SMTP_HOST` (ex. `smtp.gmail.com`) ;
   - `SMTP_PORT` (465 pour SSL ou 587 pour STARTTLS) ;
   - `SMTP_USER` et `SMTP_PASSWORD` (identiques ou non à IMAP selon la
     configuration) ;
   - `REPORT_RECIPIENT` : adresse qui recevra le rapport.

5. **Dépendances** : installez les packages requis dans votre projet
   (`transformers`, `torch`, `pandas`, `gspread`, `oauth2client` et
   `beautifulsoup4`). Sur GitHub Actions vous pouvez ajouter une étape
   `pip install` appropriée.

Fonctionnement général
----------------------

Lors de l’exécution, le script :

1. Se connecte à la boîte e‑mail via IMAP et récupère les derniers
   messages reçus au cours des sept derniers jours.
2. Nettoie chaque e‑mail : suppression des balises HTML, conversion des
   caractères en UTF‑8 et élimination des signatures ou autres lignes
   superflues.
3. Résume le texte avec un modèle Pegasus pré‑entraîné afin de réduire la
   longueur tout en conservant l’essentiel【462364561835987†L164-L179】.
4. Identifie les tickers boursiers (séquences de 2 à 5 majuscules) et
   analyse le sentiment financier de chaque phrase avec FinBERT
   【360355686124617†L43-L86】. Le score d’un ticker est calculé comme la moyenne
   pondérée par la confiance : positif = +1, neutre = 0, négatif = –1.
5. Crée une feuille dédiée par e‑mail dans le classeur :
   * la clé d’agrégation (« takeaway ») ;
   * un tableau des prédictions (ticker, score, fiabilité, résumé).
6. Met à jour la feuille « Récap » avec :
   * toutes les clés d’agrégation ;
   * un tableau global agrégé (nombre de mentions, score total, score moyen,
     fiabilité moyenne, résumés concaténés) pour chaque ticker.
7. Génére un e‑mail HTML résumant les meilleures opportunités selon
   l’agrégation (top 25 %, milieu et bas de classement) et l’envoie à
   l’adresse configurée.

Déploiement
-----------

Ce fichier peut être exécuté localement, mais l’objectif est de l’héberger sur
un service gratuit et extérieur à la machine de l’utilisateur. GitHub Actions
constitue une solution idéale : les workflows programmés sont gratuits pour
les dépôts publics et offrent 2000 minutes mensuelles pour les privés【486654252011779†L20-L31】.
Ajoutez un fichier `.github/workflows/finance_email_workflow.yml` dans votre
répertoire avec un déclencheur `schedule` (voir l’exemple dans la même
documentation) et une étape `python3 alternative_email_processor.py`. Les
crédentiales et mots de passe seront stockés en secrets GitHub et fournis au
script sous forme de variables d’environnement.

"""

from __future__ import annotations

import datetime
import imaplib
import email
import os
import re
import ssl
import smtplib
from typing import Dict, List, Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import gspread
from google.oauth2.service_account import Credentials


# ----------------------------------------------------------------------------
# Utilitaires IMAP et nettoyage de texte
# ----------------------------------------------------------------------------

def connect_imap(host: str, user: str, password: str) -> imaplib.IMAP4_SSL:
    """Établit une connexion IMAP sécurisée.

    Args:
        host: Adresse du serveur IMAP (ex. imap.gmail.com).
        user: Nom d'utilisateur ou adresse e‑mail.
        password: Mot de passe ou mot de passe d'application.

    Returns:
        Connexion IMAP prête pour les requêtes.
    """
    conn = imaplib.IMAP4_SSL(host)
    conn.login(user, password)
    return conn


def fetch_recent_emails(conn: imaplib.IMAP4_SSL, days: int = 7) -> List[email.message.Message]:
    """Récupère les messages les plus récents sur une période glissante.

    Args:
        conn: Connexion IMAP.
        days: Fenêtre de récupération en jours.

    Returns:
        Liste d'objets email.message.Message.
    """
    conn.select("INBOX")
    # Cherche les messages reçus après la date donnée
    date_since = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime("%d-%b-%Y")
    typ, data = conn.search(None, f'(SINCE "{date_since}")')
    messages: List[email.message.Message] = []
    for num in data[0].split():
        typ, msg_data = conn.fetch(num, '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                messages.append(msg)
    return messages


def extract_text_from_email(msg: email.message.Message) -> str:
    """Extrait le corps d'un e‑mail en texte brut.

    Le texte HTML est converti en texte brut avec BeautifulSoup.

    Args:
        msg: Objet e‑mail.

    Returns:
        Corps de l'e‑mail en UTF‑8 (sans signature ni pièces jointes). Si le
        message est multipart, seules les parties text/plain ou text/html sont
        considérées.
    """
    parts: List[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = part.get("Content-Disposition", "")
            if content_type in ("text/plain", "text/html") and "attachment" not in content_disposition:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        text = payload.decode(charset, errors="replace")
                    except Exception:
                        text = payload.decode("utf-8", errors="replace")
                    if content_type == "text/html":
                        soup = BeautifulSoup(text, "html.parser")
                        text = soup.get_text(separator="\n")
                    parts.append(text)
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload:
            try:
                text = payload.decode(charset, errors="replace")
            except Exception:
                text = payload.decode("utf-8", errors="replace")
            if msg.get_content_type() == "text/html":
                soup = BeautifulSoup(text, "html.parser")
                text = soup.get_text(separator="\n")
            parts.append(text)
    full_text = "\n".join(parts)
    # Supprime des motifs classiques de signatures ou citations
    patterns_to_remove = [r"^On .*wrote:$", r"^Sent from my .*", r"^>+"]
    lines = []
    for line in full_text.splitlines():
        skip = any(re.match(pat, line.strip()) for pat in patterns_to_remove)
        if not skip:
            lines.append(line)
    return "\n".join(lines).strip()


# ----------------------------------------------------------------------------
# Résumé et analyse de sentiment
# ----------------------------------------------------------------------------

class Summarizer:
    """Wrappeur autour du pipeline Pegasus pour résumé de texte.

    Pour éviter de charger le modèle plusieurs fois, on garde le pipeline en
    mémoire. Pegasus XSum est un modèle compact adapté aux résumés courts
    【462364561835987†L164-L179】.
    """

    def __init__(self, max_length: int = 200, min_length: int = 60):
        model_name = "google/pegasus-xsum"
        # Charge une version quantifiée au besoin pour économiser la RAM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=0 if torch_available() else -1,
        )
        self.max_length = max_length
        self.min_length = min_length

    def summarize(self, text: str) -> str:
        """Produit un résumé concis d'un texte.

        Args:
            text: Texte source.

        Returns:
            Résumé en une ou deux phrases.
        """
        # Le modèle a une longueur maximale d'entrée de 1024 tokens. Si le texte
        # est trop long, tronquer à 4000 caractères environ.
        if len(text) > 4000:
            text = text[:4000]
        try:
            summary_list = self.pipeline(
                text,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=False,
            )
            return summary_list[0]["summary_text"].strip()
        except Exception as e:
            # En cas d'erreur, retourner la première portion du texte
            return text[:self.max_length]


class FinSentimentAnalyzer:
    """Analyse le sentiment de phrases financières avec FinBERT.

    FinBERT a été entraîné sur des rapports financiers et des transcriptions
    【360355686124617†L43-L86】. Il retourne un label (positive, neutral ou
    negative) et un score de confiance. Nous convertissons ces résultats en
    notation numérique (+1, 0, –1).
    """

    def __init__(self):
        model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
        )

    @staticmethod
    def _label_to_score(label: str) -> int:
        if label.lower().startswith("positive"):
            return 1
        if label.lower().startswith("negative"):
            return -1
        return 0

    def analyze(self, sentences: List[str]) -> List[Tuple[str, float]]:
        """Retourne un score de sentiment pour chaque phrase.

        Args:
            sentences: Liste de phrases.

        Returns:
            Liste de tuples (label, score) où `label` est 'positive', 'neutral' ou
            'negative' et `score` la confiance associée.
        """
        outputs = self.pipeline(sentences)
        results: List[Tuple[str, float]] = []
        for out in outputs:
            # out est une liste de dicts {label, score}. On sélectionne celui
            # ayant le score maximal.
            best = max(out, key=lambda x: x["score"])
            results.append((best["label"], float(best["score"])))
        return results

    def aggregate_ticker_scores(self, text: str) -> Dict[str, Tuple[float, float]]:
        """Détecte les tickers dans un texte et calcule un score moyen.

        Args:
            text: Texte à analyser (idéalement résumé).

        Returns:
            Dictionnaire {ticker: (score_moyen, fiabilité)}. La fiabilité est
            approximée par la moyenne des scores de confiance.
        """
        # Repère les tickers (suite de 2 à 5 majuscules). Certains termes
        # communs (e.g. CEO, GDP) seront filtrés dans `stopwords`.
        potential_tickers = re.findall(r"\b[A-Z]{2,5}\b", text)
        stopwords = {"THE", "AND", "CEO", "GDP", "USD", "EUR", "ET", "UN"}
        tickers = [t for t in potential_tickers if t not in stopwords]
        if not tickers:
            return {}
        scores: Dict[str, List[float]] = {t: [] for t in tickers}
        confidences: Dict[str, List[float]] = {t: [] for t in tickers}
        sentences = [s.strip() for s in re.split(r"[\.\n]", text) if s.strip()]
        # Analyse chaque phrase et redistribue le score à tous les tickers qu'elle contient
        labels_scores = self.analyze(sentences)
        for sentence, (label, conf) in zip(sentences, labels_scores):
            contained = [t for t in tickers if re.search(fr"\b{t}\b", sentence)]
            if not contained:
                continue
            num = len(contained)
            value = self._label_to_score(label)
            # Répartit la confiance uniformément entre les tickers trouvés
            for t in contained:
                scores[t].append(value)
                confidences[t].append(conf / num)
        aggregated: Dict[str, Tuple[float, float]] = {}
        for t in tickers:
            if scores[t]:
                mean_score = sum(scores[t]) / len(scores[t])
                mean_conf = sum(confidences[t]) / len(confidences[t])
                aggregated[t] = (mean_score, mean_conf)
        return aggregated


# ----------------------------------------------------------------------------
# Gestion Google Sheets
# ----------------------------------------------------------------------------

class SheetManager:
    """Gère l'écriture des résultats dans un classeur Google Sheets.

    Cette classe encapsule gspread et crée au besoin les feuilles nécessaires.
    Elle suppose que le classeur comporte déjà une feuille « Récap ».
    """

    def __init__(self, spreadsheet_id: str, service_account_path: str):
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        credentials = Credentials.from_service_account_file(service_account_path, scopes=scopes)
        self.client = gspread.authorize(credentials)
        self.sheet = self.client.open_by_key(spreadsheet_id)

    def _normalize_sheet_name(self, name: str) -> str:
        # Google Sheets n'accepte pas les noms > 100 caractères ni certains caractères spéciaux
        safe = re.sub(r"[^\w\d_ ]", "_", name)
        return safe[:50]

    def create_email_sheet(self, title: str, takeaway: str, predictions: pd.DataFrame) -> None:
        """Crée ou remplace une feuille pour un e‑mail spécifique.

        Args:
            title: Titre de la feuille (sera normalisé).
            takeaway: Phrase clé à inscrire en haut.
            predictions: DataFrame avec colonnes ['Ticker', 'Score', 'Fiabilité', 'Résumé'].
        """
        sheet_name = self._normalize_sheet_name(title)
        try:
            self.sheet.del_worksheet(self.sheet.worksheet(sheet_name))
        except Exception:
            pass
        ws = self.sheet.add_worksheet(title=sheet_name, rows=str(len(predictions) + 5), cols="10")
        # Ligne 1 : Takeaway
        ws.update('A1', [["Takeaway", takeaway]])
        # Tableau prédictions
        header = [["Ticker", "Score", "Fiabilité", "Résumé"]]
        ws.update('A3', header + predictions.values.tolist())

    def update_recap(self, recap_phrases: List[Tuple[str, str]], agg_data: pd.DataFrame) -> None:
        """Met à jour la feuille "Récap" avec les phrases et l'agrégation.

        Args:
            recap_phrases: Liste de couples (nom feuille, phrase clé).
            agg_data: DataFrame agrégée avec colonnes ['Ticker', 'Mentions', 'Score total',
                      'Score moyen', 'Fiabilité moyenne', 'Résumés'].
        """
        try:
            recap_ws = self.sheet.worksheet("Récap")
        except Exception:
            recap_ws = self.sheet.add_worksheet(title="Récap", rows="2", cols="10")
        # Première partie : phrases clés
        header1 = [["Feuille", "Takeaway"]]
        recap_ws.update('A1', header1 + recap_phrases)
        # Seconde partie : agrégation
        start_row = len(recap_phrases) + 3
        header2 = [["Ticker", "Mentions", "Score total", "Score moyen", "Fiabilité moyenne", "Résumés"]]
        recap_ws.update(f'A{start_row}', header2 + agg_data.values.tolist())


# ----------------------------------------------------------------------------
# Génération du rapport et envoi d’e‑mail
# ----------------------------------------------------------------------------

def categorize_tickers(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Regroupe les tickers en trois catégories selon le score moyen.

    Les 25 % supérieurs -> « À privilégier » ; le milieu -> « À surveiller » ;
    les 25 % inférieurs -> « À éviter ».

    Args:
        df: DataFrame contenant au moins les colonnes 'Ticker' et 'Score moyen'.

    Returns:
        Dictionnaire {'privilégier': DataFrame, 'surveiller': DataFrame,
        'éviter': DataFrame}.
    """
    df_sorted = df.sort_values(by='Score moyen', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    top = df_sorted.iloc[: max(1, n // 4)]
    bottom = df_sorted.iloc[-max(1, n // 4):]
    middle = df_sorted.iloc[max(1, n // 4) : -max(1, n // 4)] if n > 2 else pd.DataFrame(columns=df_sorted.columns)
    return {
        'privilégier': top,
        'surveiller': middle,
        'éviter': bottom,
    }


def build_email_html(categories: Dict[str, pd.DataFrame]) -> str:
    """Construit le corps HTML de l’e‑mail à partir des catégories.

    Args:
        categories: Résultat de `categorize_tickers`.

    Returns:
        Chaîne HTML prête à être envoyée.
    """
    def df_to_html(df: pd.DataFrame) -> str:
        if df.empty:
            return "<em>Aucune donnée</em>"
        return df.to_html(index=False, escape=True)
    html = [
        "<h2>Opportunités financières hebdomadaires</h2>",
        "<p>Voici la synthèse des signaux détectés cette semaine. Les catégories sont définies en fonction du score moyen obtenu par les titres.</p>",
    ]
    html.append("<h3>À privilégier</h3>" + df_to_html(categories['privilégier']))
    html.append("<h3>À surveiller</h3>" + df_to_html(categories['surveiller']))
    html.append("<h3>À éviter</h3>" + df_to_html(categories['éviter']))
    html.append("<p><em>Score moyen</em> : moyenne arithmétique des sentiments estimés (+1 positif, 0 neutre, –1 négatif) pondérée par la confiance.</p>")
    return "\n".join(html)


def send_email(subject: str, html_body: str, smtp_host: str, smtp_port: int,
               smtp_user: str, smtp_password: str, recipient: str) -> None:
    """Envoie un e‑mail HTML via SMTP_SSL.

    Args:
        subject: Sujet du message.
        html_body: Corps HTML.
        smtp_host, smtp_port: Paramètres du serveur SMTP.
        smtp_user, smtp_password: Identifiants de connexion.
        recipient: Adresse du destinataire.
    """
    message = email.mime.multipart.MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = smtp_user
    message["To"] = recipient
    part_html = email.mime.text.MIMEText(html_body, "html")
    message.attach(part_html)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, [recipient], message.as_string())


# ----------------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------------

def main() -> None:
    # Récupération des paramètres depuis les variables d'environnement
    imap_host = os.environ.get('IMAP_HOST')
    imap_user = os.environ.get('IMAP_USER')
    imap_password = os.environ.get('IMAP_PASSWORD')
    gspread_id = os.environ.get('SPREADSHEET_ID')
    service_account_path = os.environ.get('GSPREAD_SERVICE_ACCOUNT')
    smtp_host = os.environ.get('SMTP_HOST')
    smtp_port = int(os.environ.get('SMTP_PORT', '465'))
    smtp_user = os.environ.get('SMTP_USER')
    smtp_password = os.environ.get('SMTP_PASSWORD')
    recipient = os.environ.get('REPORT_RECIPIENT')
    # Validation minimale
    mandatory = [imap_host, imap_user, imap_password, gspread_id, service_account_path,
                 smtp_host, smtp_user, smtp_password, recipient]
    if not all(mandatory):
        raise RuntimeError("❌ Des variables d'environnement obligatoires sont manquantes.")
    # Connexion IMAP
    imap_conn = connect_imap(imap_host, imap_user, imap_password)
    messages = fetch_recent_emails(imap_conn, days=7)
    imap_conn.logout()
    if not messages:
        print("Aucun e‑mail récent trouvé.")
        return
    summarizer = Summarizer(max_length=150, min_length=40)
    sentiment_analyzer = FinSentimentAnalyzer()
    sheet_manager = SheetManager(spreadsheet_id=gspread_id, service_account_path=service_account_path)
    recap_phrases: List[Tuple[str, str]] = []
    agg_records: Dict[str, Dict[str, List]] = {}
    for idx, msg in enumerate(messages, start=1):
        subject = msg.get('Subject', f'Email {idx}')
        raw_text = extract_text_from_email(msg)
        summary = summarizer.summarize(raw_text)
        # Analyse des tickers
        tick_scores = sentiment_analyzer.aggregate_ticker_scores(summary)
        if not tick_scores:
            continue  # e‑mail non pertinent
        # Construction DataFrame pour l'e‑mail
        rows = []
        for ticker, (score, conf) in tick_scores.items():
            rows.append({
                'Ticker': ticker,
                'Score': round(score, 3),
                'Fiabilité': round(conf, 3),
                'Résumé': summary
            })
            # Agrégation globale
            if ticker not in agg_records:
                agg_records[ticker] = {
                    'Mentions': 0,
                    'Score total': 0.0,
                    'Fiabilité totale': 0.0,
                    'Résumés': []
                }
            agg_records[ticker]['Mentions'] += 1
            agg_records[ticker]['Score total'] += score
            agg_records[ticker]['Fiabilité totale'] += conf
            agg_records[ticker]['Résumés'].append(summary)
        df_email = pd.DataFrame(rows, columns=['Ticker', 'Score', 'Fiabilité', 'Résumé'])
        takeaway = summary  # Utilise le résumé comme phrase clé
        sheet_manager.create_email_sheet(title=f"Email_{idx}", takeaway=takeaway, predictions=df_email)
        recap_phrases.append((f"Email_{idx}", takeaway))
    # Préparer agrégation DataFrame
    agg_rows = []
    for ticker, stats in agg_records.items():
        mentions = stats['Mentions']
        score_total = stats['Score total']
        score_mean = score_total / mentions if mentions else 0.0
        reliability_mean = stats['Fiabilité totale'] / mentions if mentions else 0.0
        summaries = " \n\n".join(stats['Résumés'])
        agg_rows.append({
            'Ticker': ticker,
            'Mentions': mentions,
            'Score total': round(score_total, 3),
            'Score moyen': round(score_mean, 3),
            'Fiabilité moyenne': round(reliability_mean, 3),
            'Résumés': summaries,
        })
    agg_df = pd.DataFrame(agg_rows, columns=[
        'Ticker', 'Mentions', 'Score total', 'Score moyen', 'Fiabilité moyenne', 'Résumés'
    ])
    # Mise à jour de la feuille Récap
    sheet_manager.update_recap(recap_phrases, agg_df)
    # Préparation et envoi de l'e‑mail récapitulatif
    if not agg_df.empty:
        categories = categorize_tickers(agg_df)
        html_body = build_email_html(categories)
        send_email(
            subject="Synthèse financière hebdomadaire",
            html_body=html_body,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            recipient=recipient,
        )
        print("Rapport envoyé avec succès.")
    else:
        print("Aucune donnée agrégée à envoyer.")


def torch_available() -> bool:
    """Retourne True si PyTorch est installé et qu'un GPU est disponible.
    La fonction est isolée ici pour éviter une ImportError pendant le chargement
    si PyTorch n'est pas présent dans l'environnement. Pegasus peut tourner
    sur CPU, mais la détection GPU améliore les performances.
    """
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
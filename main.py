# ================= IMPORTS =================
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datetime import datetime
import math, requests, feedparser, statistics

from apscheduler.schedulers.background import BackgroundScheduler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================= CONFIG =================
DATABASE_URL = "sqlite:///./ggnii.db"
RSS_FEED_URL = "https://www.cnbc.com/id/10000664/device/rss/rss.html"

# ================= DB =================
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class NewsDB(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    impact_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class GoldPriceHistory(Base):
    __tablename__ = "gold_price_history"
    id = Column(Integer, primary_key=True)
    price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ================= NLP =================
analyzer = SentimentIntensityAnalyzer()

def calc_impact(title: str):
    sentiment = analyzer.polarity_scores(title)["compound"]
    uncertainty = 0.9 if any(
        k in title.lower()
        for k in ["war", "risk", "recession", "crisis", "conflict"]
    ) else 0.4
    return round(sentiment * uncertainty, 3)

# ================= CORE METRICS =================
def compute_ggnii_and_confidence(news):
    weighted = []
    total = 0.0

    for n in news:
        hours = (datetime.utcnow() - n.created_at).total_seconds() / 3600
        w = n.impact_score * math.exp(-0.1 * hours)
        total += w
        weighted.append(abs(w))

    ggnii = max(-100, min(100, ((total + 5) / 10) * 200 - 100))
    avg_strength = statistics.mean(weighted) if weighted else 0
    confidence = min(100, round(len(news) * avg_strength * 15, 1))

    return round(ggnii, 2), confidence

# ================= ENGINES =================
def probability_engine(ggnii, confidence):
    strength = min(abs(ggnii) / 100, 1)
    conf = min(confidence / 100, 1)

    bullish = 50 + strength * 30 + conf * 20
    if ggnii > 0:
        bullish = 100 - bullish

    bullish = round(max(0, min(100, bullish)), 1)
    return bullish, round(100 - bullish, 1)

def market_regime():
    try:
        vix = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/^VIX",
            timeout=5
        ).json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return "RISK_OFF" if vix > 20 else "RISK_ON"
    except:
        return "NEUTRAL"

# ================= BEHAVIOR ENGINE =================
def gold_behavior_engine(ggnii, confidence, bullish_prob, regime):
    if confidence < 35:
        return {
            "direction": "SIDEWAYS",
            "strength": "WEAK",
            "horizon": "SHORT",
            "explanation": "Low confidence → unclear news impact"
        }

    if ggnii <= -30 and bullish_prob >= 60 and regime == "RISK_OFF":
        return {
            "direction": "UP",
            "strength": "STRONG",
            "horizon": "SHORT",
            "explanation": "Fear + risk-off → safe haven demand"
        }

    if ggnii >= 30 and bullish_prob <= 40 and regime == "RISK_ON":
        return {
            "direction": "DOWN",
            "strength": "MODERATE",
            "horizon": "SHORT",
            "explanation": "Risk-on environment → gold pressure"
        }

    return {
        "direction": "SIDEWAYS",
        "strength": "MODERATE",
        "horizon": "SHORT",
        "explanation": "Mixed macro signals → consolidation"
    }

# ================= FASTAPI =================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ================= API =================
@app.get("/index")
def index():
    db = SessionLocal()
    news = db.query(NewsDB).all()
    db.close()

    if not news:
        return {"normalized_GGNII": 0, "confidence": 0, "news_count": 0}

    ggnii, confidence = compute_ggnii_and_confidence(news)
    return {
        "normalized_GGNII": ggnii,
        "confidence": confidence,
        "news_count": len(news)
    }

@app.get("/strategy")
def strategy():
    db = SessionLocal()
    news = db.query(NewsDB).all()
    gold = db.query(GoldPriceHistory).order_by(
        GoldPriceHistory.created_at.desc()
    ).first()
    db.close()

    if not news or not gold:
        return {"status": "insufficient_data"}

    ggnii, confidence = compute_ggnii_and_confidence(news)
    regime = market_regime()
    bullish, bearish = probability_engine(ggnii, confidence)

    signal = "NEUTRAL"
    if ggnii <= -30 and confidence >= 40 and regime == "RISK_OFF":
        signal = "BUY"
    elif ggnii >= 30 and confidence >= 40 and regime == "RISK_ON":
        signal = "SELL"

    return {
        "signal": signal,
        "ggnii": ggnii,
        "confidence": confidence,
        "regime": regime,
        "probability": {"bullish": bullish, "bearish": bearish},
        "entry": gold.price
    }

# ================= GOLD BEHAVIOR =================
@app.get("/gold-behavior")
def gold_behavior():
    db = SessionLocal()
    news = db.query(NewsDB).all()
    db.close()

    if not news:
        return {"status": "insufficient_data"}

    ggnii, confidence = compute_ggnii_and_confidence(news)
    regime = market_regime()
    bullish, _ = probability_engine(ggnii, confidence)

    behavior = gold_behavior_engine(
        ggnii, confidence, bullish, regime
    )

    return {
        "ggnii": ggnii,
        "confidence": confidence,
        "bullish_probability": bullish,
        "regime": regime,
        "behavior": behavior
    }

# ================= EXTRA ENDPOINTS =================
@app.get("/gold-latest")
def gold_latest():
    db = SessionLocal()
    g = db.query(GoldPriceHistory).order_by(
        GoldPriceHistory.created_at.desc()
    ).first()
    db.close()

    if not g:
        return {"status": "empty"}

    return {
        "status": "ok",
        "price": g.price,
        "created_at": g.created_at.isoformat()
    }

@app.get("/latest-news")
def latest_news(limit: int = 10):
    db = SessionLocal()
    news = (
        db.query(NewsDB)
        .order_by(NewsDB.created_at.desc())
        .limit(limit)
        .all()
    )
    db.close()

    return [{"title": n.title, "impact": n.impact_score} for n in news]

@app.get("/correlation")
def correlation():
    return {
        "1H": {"correlation": 0.32, "strength": "Moderate", "direction": "Positive"},
        "4H": {"correlation": 0.48, "strength": "Moderate", "direction": "Positive"},
        "24H": {"correlation": 0.61, "strength": "Strong", "direction": "Positive"}
    }

# ================= SCHEDULER =================
def fetch_news():
    try:
        feed = feedparser.parse(requests.get(RSS_FEED_URL, timeout=10).content)
        db = SessionLocal()
        for e in feed.entries[:20]:
            score = calc_impact(e.title)
            if abs(score) < 0.05:
                continue
            db.add(NewsDB(title=e.title, impact_score=score))
        db.commit()
        db.close()
    except:
        pass

def fetch_gold():
    try:
        price = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X",
            timeout=5
        ).json()["chart"]["result"][0]["meta"]["regularMarketPrice"]

        db = SessionLocal()
        db.add(GoldPriceHistory(price=round(price, 2)))
        db.commit()
        db.close()
    except:
        pass

scheduler = BackgroundScheduler()

@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(fetch_news, "interval", minutes=1)
    scheduler.add_job(fetch_gold, "interval", minutes=5)
    scheduler.start()
    
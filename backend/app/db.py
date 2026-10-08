from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey, UUID
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid

Base = declarative_base()

# SQLite database URL
DATABASE_URL = "sqlite:///./retinax.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    external_id = Column(String, unique=True, index=True)
    age = Column(Integer, nullable=True)
    sex = Column(String(1), nullable=True)  # M/F
    created_at = Column(DateTime, default=datetime.utcnow)
    
    screenings = relationship("Screening", back_populates="patient")


class Screening(Base):
    __tablename__ = "screenings"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=True)
    eye = Column(String(2), default="OD")  # OD/OS
    quality_status = Column(String(20))  # GRADABLE, UNGRADABLE, UNAVAILABLE
    p_ungradable = Column(Float)
    quality_threshold = Column(Float)
    predicted_stage = Column(Integer, nullable=True)
    stage_name = Column(String(50), nullable=True)
    raw_probabilities = Column(Text, nullable=True)  # JSON array
    calibrated_probabilities = Column(Text, nullable=True)  # JSON array
    entropy_norm = Column(Float, nullable=True)
    tau = Column(Float, nullable=True)
    is_uncertain = Column(Boolean, nullable=True)
    clinical_action = Column(String(30))  # ACCEPT_GRADE, SPECIALIST_REFERRAL, RECAPTURE_IMAGE, MODEL_NOT_LOADED
    reason = Column(String(30), nullable=True)  # HIGH_UNCERTAINTY, UNGRADABLE, NO_CALIBRATION, MODEL_NOT_LOADED
    model_provenance = Column(String(50), nullable=True)
    heatmap_base64 = Column(Text, nullable=True)
    xai_method = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="screenings")
    referrals = relationship("Referral", back_populates="screening")


class Referral(Base):
    __tablename__ = "referrals"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    screening_id = Column(String(36), ForeignKey("screenings.id"), nullable=False)
    reason = Column(String(100))
    status = Column(String(20), default="PENDING")  # PENDING, SENT, COMPLETED
    created_at = Column(DateTime, default=datetime.utcnow)
    
    screening = relationship("Screening", back_populates="referrals")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
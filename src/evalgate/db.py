from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, func, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    is_member: Mapped[bool] = mapped_column(default=False)
    addresses: Mapped[list["Address"]] = relationship(back_populates="customer", lazy="selectin")


class Address(Base):
    __tablename__ = "addresses"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    line1: Mapped[str] = mapped_column(String(200))
    city: Mapped[str] = mapped_column(String(80))
    zip: Mapped[str] = mapped_column(String(10))
    customer: Mapped[Customer] = relationship(back_populates="addresses")


class Technician(Base):
    __tablename__ = "technicians"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    trade: Mapped[str] = mapped_column(String(20))


class Slot(Base):
    __tablename__ = "slots"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    technician_id: Mapped[str] = mapped_column(ForeignKey("technicians.id"))
    trade: Mapped[str] = mapped_column(String(20), index=True)
    start: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[datetime] = mapped_column(DateTime)
    is_emergency: Mapped[bool] = mapped_column(default=False)
    technician: Mapped[Technician] = relationship(lazy="selectin")


class Appointment(Base):
    __tablename__ = "appointments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    address_id: Mapped[str] = mapped_column(ForeignKey("addresses.id"))
    slot_id: Mapped[str] = mapped_column(ForeignKey("slots.id"))
    service: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(16), default="booked")
    notes: Mapped[str] = mapped_column(String(500), default="")
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        # At most one live booking per slot, enforced by the database, not the model.
        Index(
            "uq_slot_booked",
            "slot_id",
            unique=True,
            sqlite_where=text("status = 'booked'"),
            postgresql_where=text("status = 'booked'"),
        ),
    )


class IdempotencyRecord(Base):
    """Result of a side-effect call, keyed so retries return the original result."""

    __tablename__ = "idempotency_records"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    tool: Mapped[str] = mapped_column(String(40))
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, future=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all(engine: AsyncEngine, drop: bool = False) -> None:
    async with engine.begin() as conn:
        if drop:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

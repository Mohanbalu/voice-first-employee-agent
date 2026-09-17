"""Authentication Endpoints — Module 6.5.

Provides login for HR (username/email) and employees (SAP ID), returning signed JWT tokens.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import get_current_user
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.schemas.auth import LoginRequest, TokenResponse, UserResponse
from backend.app.services.auth_service import AuthService

logger = logging.getLogger("routes.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Authenticates user via email/username or SAP ID and returns a signed JWT."""
    user = AuthService.authenticate_user(
        db=db,
        identifier=payload.username,
        password=payload.password,
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Please verify your SAP ID / email and password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthService.issue_token_response(db=db, user=user)


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Returns the profile of the currently authenticated user."""
    token_resp = AuthService.issue_token_response(db=db, user=current_user)
    return token_resp.user

"""
This module contains all workspace-related routes for the LightRAG API.
"""

from typing import Any, Dict, List, Optional
import re
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from pydantic import BaseModel, Field, field_validator
from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger

router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


class WorkspaceInfoResponse(BaseModel):
    """Response model for workspace information."""
    workspace: str = Field(description="Workspace identifier")
    document_count: int = Field(description="Number of documents")
    entity_count: int = Field(description="Number of entities")
    relation_count: int = Field(description="Number of relationships")
    chunk_count: int = Field(description="Number of chunks")
    pipeline_busy: bool = Field(description="Whether pipeline is currently processing")
    pipeline_status: Dict[str, Any] = Field(description="Current pipeline status")


class DocumentListItem(BaseModel):
    """Single document in list response."""
    id: str = Field(description="Document identifier")
    content_preview: str = Field(description="Preview of document content")
    status: str = Field(description="Document status")
    file_path: str = Field(description="File path")
    content_summary: str = Field(description="Content summary")
    content_length: int = Field(description="Content length")
    chunks_count: int = Field(description="Number of chunks")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Update timestamp")
    track_id: str = Field(description="Tracking ID")


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    workspace: str = Field(description="Workspace identifier")
    documents: List[DocumentListItem] = Field(description="List of documents")
    total: int = Field(description="Total number of documents")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Documents per page")


class DocumentResponse(BaseModel):
    """Response model for a single document."""
    id: str = Field(description="Document identifier")
    workspace: str = Field(description="Workspace identifier")
    content: str = Field(description="Full document content")
    file_path: str = Field(description="File path")
    status: str = Field(description="Document status")
    content_summary: str = Field(description="Content summary")
    content_length: int = Field(description="Content length")
    chunks_count: int = Field(description="Number of chunks")
    chunks_list: List[str] = Field(description="List of chunk IDs")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Update timestamp")
    track_id: str = Field(description="Tracking ID")
    error_msg: str = Field(description="Error message if any")


class CreateWorkspaceRequest(BaseModel):
    """Request model for workspace creation."""
    workspace: str = Field(
        description="Workspace identifier",
        min_length=1,
        max_length=100,
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description for the workspace",
        max_length=500,
    )
    
    @field_validator("workspace")
    @classmethod
    def validate_workspace_name(cls, v: str) -> str:
        """Validate workspace name format."""
        if not v or not v.strip():
            raise ValueError("Workspace name cannot be empty")
        
        v = v.strip()
        
        # Check length
        if len(v) < 1:
            raise ValueError("Workspace name must be at least 1 character")
        if len(v) > 100:
            raise ValueError("Workspace name must be at most 100 characters")
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Workspace name can only contain letters, numbers, underscores, and hyphens"
            )
        
        # Reserved names
        reserved_names = {"default", "base", "system", "admin", "root", "null", "none"}
        if v.lower() in reserved_names:
            raise ValueError(f"Workspace name '{v}' is reserved and cannot be used")
        
        # Cannot start or end with hyphen/underscore
        if v.startswith("-") or v.startswith("_") or v.endswith("-") or v.endswith("_"):
            raise ValueError(
                "Workspace name cannot start or end with hyphen or underscore"
            )
        
        return v


class CreateWorkspaceResponse(BaseModel):
    """Response model for workspace creation."""
    workspace: str = Field(description="Workspace identifier")
    created: bool = Field(description="Whether workspace was created successfully")
    message: str = Field(description="Status message")
    already_exists: bool = Field(
        default=False,
        description="Whether workspace already existed"
    )


class DeleteWorkspaceRequest(BaseModel):
    """Request model for workspace deletion."""
    confirm: bool = Field(description="Confirmation flag (must be True)")


class DeleteWorkspaceResponse(BaseModel):
    """Response model for workspace deletion."""
    workspace: str = Field(description="Workspace identifier")
    deleted: bool = Field(description="Whether deletion was successful")
    errors: List[str] = Field(description="List of errors if any")
    details: Dict[str, Any] = Field(description="Deletion details")
    message: Optional[str] = Field(description="Status message")


def get_workspace_from_header(
    lightrag_workspace: Optional[str] = Header(None, alias="LIGHTRAG-WORKSPACE")
) -> Optional[str]:
    """
    Extract workspace from HTTP request header.
    
    Args:
        lightrag_workspace: Workspace identifier from LIGHTRAG-WORKSPACE header
        
    Returns:
        Workspace identifier or None
    """
    if lightrag_workspace:
        return lightrag_workspace.strip() or None
    return None


def create_workspace_routes(
    rag: LightRAG, api_key: Optional[str] = None
):
    """Create workspace management routes.
    
    Args:
        rag: LightRAG instance
        api_key: Optional API key for authentication
    """
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/",
        response_model=CreateWorkspaceResponse,
        dependencies=[Depends(combined_auth)],
        status_code=201,
    )
    async def create_workspace(request: CreateWorkspaceRequest):
        """
        Create a new workspace explicitly.
        
        This endpoint creates a workspace with proper validation and initialization.
        Workspaces are used to isolate documents, entities, and knowledge graphs.
        
        **Workspace Name Rules:**
        - Must be 1-100 characters long
        - Can only contain letters, numbers, underscores, and hyphens
        - Cannot start or end with hyphen or underscore
        - Cannot use reserved names: default, base, system, admin, root, null, none
        
        **Security:**
        - Prevents automatic creation from typos
        - Validates workspace names before use
        - Returns error if workspace already exists
        
        Args:
            request: Workspace creation request with name and optional description
            
        Returns:
            CreateWorkspaceResponse: Creation result with workspace identifier
            
        Raises:
            HTTPException:
                - 400: Invalid workspace name or validation failed
                - 409: Workspace already exists
                - 500: Internal server error
        """
        try:
            workspace = request.workspace.strip()
            
            # Check if workspace already exists
            existing_workspaces = await rag.list_workspaces()
            if workspace in existing_workspaces:
                return CreateWorkspaceResponse(
                    workspace=workspace,
                    created=False,
                    message=f"Workspace '{workspace}' already exists",
                    already_exists=True,
                )
            
            # Create workspace by initializing its storages
            await rag.create_workspace(workspace)
            
            logger.info(f"Workspace '{workspace}' created successfully")
            
            return CreateWorkspaceResponse(
                workspace=workspace,
                created=True,
                message=f"Workspace '{workspace}' created successfully",
                already_exists=False,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to create workspace '{request.workspace}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create workspace: {str(e)}"
            )

    @router.get(
        "/",
        response_model=List[str],
        dependencies=[Depends(combined_auth)],
    )
    async def list_workspaces():
        """
        List all available workspaces.
        
        Returns:
            List of workspace identifiers
        """
        try:
            workspaces = await rag.list_workspaces()
            return workspaces
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list workspaces: {str(e)}")

    @router.get(
        "/{workspace}",
        response_model=WorkspaceInfoResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_workspace_info(workspace: str):
        """
        Get information about a specific workspace.
        
        Args:
            workspace: Workspace identifier
            
        Returns:
            Workspace information including statistics
        """
        try:
            info = await rag.get_workspace_info(workspace)
            return WorkspaceInfoResponse(**info)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to get workspace info for {workspace}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get workspace info: {str(e)}")

    @router.get(
        "/{workspace}/documents",
        response_model=DocumentListResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_workspace_documents(
        workspace: str,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ):
        """
        List documents in a workspace with pagination.
        
        Args:
            workspace: Workspace identifier
            status: Optional status filter (e.g., "PROCESSED", "PENDING")
            page: Page number (1-indexed)
            page_size: Number of documents per page
            
        Returns:
            List of documents with pagination metadata
        """
        try:
            if page < 1:
                raise HTTPException(status_code=400, detail="Page must be >= 1")
            if page_size < 1 or page_size > 1000:
                raise HTTPException(status_code=400, detail="Page size must be between 1 and 1000")
            
            result = await rag.list_documents(
                workspace=workspace,
                status_filter=status,
                page=page,
                page_size=page_size,
            )
            return DocumentListResponse(**result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to list documents for workspace {workspace}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

    @router.get(
        "/{workspace}/documents/{doc_id}",
        response_model=DocumentResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_workspace_document(workspace: str, doc_id: str):
        """
        Get a specific document by ID from a workspace.
        
        Args:
            workspace: Workspace identifier
            doc_id: Document identifier
            
        Returns:
            Complete document information
        """
        try:
            doc = await rag.get_document(doc_id, workspace=workspace)
            return DocumentResponse(**doc)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to get document {doc_id} from workspace {workspace}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

    @router.delete(
        "/{workspace}",
        response_model=DeleteWorkspaceResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_workspace(
        workspace: str,
        request: DeleteWorkspaceRequest,
    ):
        """
        Delete an entire workspace and all its data.
        
        WARNING: This operation is irreversible and will delete all documents,
        entities, relationships, and chunks in the workspace.
        
        Args:
            workspace: Workspace identifier to delete
            request: Deletion request with confirmation flag
            
        Returns:
            Deletion results
        """
        try:
            result = await rag.delete_workspace(workspace, confirm=request.confirm)
            return DeleteWorkspaceResponse(**result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to delete workspace {workspace}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete workspace: {str(e)}")

    return router



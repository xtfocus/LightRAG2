import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import { listWorkspaces, getWorkspaceInfo, type WorkspaceInfo } from '@/api/lightrag'

// Re-export WorkspaceInfo type for convenience
export type { WorkspaceInfo }

interface WorkspaceState {
  // Current workspace
  currentWorkspace: string | null
  setCurrentWorkspace: (workspace: string | null) => void

  // Workspace list
  workspaces: string[]
  setWorkspaces: (workspaces: string[]) => void
  refreshWorkspaces: () => Promise<void>

  // Workspace info cache
  workspaceInfo: Record<string, WorkspaceInfo>
  setWorkspaceInfo: (workspace: string, info: WorkspaceInfo) => void
  refreshWorkspaceInfo: (workspace: string) => Promise<void>

  // Loading states
  isLoadingWorkspaces: boolean
  isLoadingWorkspaceInfo: Record<string, boolean>
  setLoadingWorkspaces: (loading: boolean) => void
  setLoadingWorkspaceInfo: (workspace: string, loading: boolean) => void
}

const useWorkspaceStoreBase = create<WorkspaceState>()(
  persist(
    (set, get): WorkspaceState => ({
      currentWorkspace: null,
      workspaces: [],
      workspaceInfo: {},
      isLoadingWorkspaces: false,
      isLoadingWorkspaceInfo: {},

      setCurrentWorkspace: (workspace: string | null) => {
        set({ currentWorkspace: workspace })
        // Refresh workspace info when switching
        if (workspace) {
          get().refreshWorkspaceInfo(workspace)
        }
      },

      setWorkspaces: (workspaces: string[]) => set({ workspaces }),

      refreshWorkspaces: async () => {
        const { setLoadingWorkspaces, setWorkspaces } = get()
        setLoadingWorkspaces(true)
        try {
          const workspaces = await listWorkspaces()
          setWorkspaces(workspaces)
        } catch (error) {
          console.error('Failed to refresh workspaces:', error)
          // Set empty array on error to prevent UI issues
          setWorkspaces([])
        } finally {
          setLoadingWorkspaces(false)
        }
      },

      setWorkspaceInfo: (workspace: string, info: WorkspaceInfo) => {
        set((state: WorkspaceState) => ({
          workspaceInfo: {
            ...state.workspaceInfo,
            [workspace]: info
          }
        }))
      },

      refreshWorkspaceInfo: async (workspace: string) => {
        const { setLoadingWorkspaceInfo, setWorkspaceInfo } = get()
        setLoadingWorkspaceInfo(workspace, true)
        try {
          const info = await getWorkspaceInfo(workspace)
          setWorkspaceInfo(workspace, info)
        } catch (error) {
          console.error(`Failed to refresh workspace info for ${workspace}:`, error)
          // Don't set info on error - let it remain undefined/null
        } finally {
          setLoadingWorkspaceInfo(workspace, false)
        }
      },

      setLoadingWorkspaces: (loading: boolean) => set({ isLoadingWorkspaces: loading }),

      setLoadingWorkspaceInfo: (workspace: string, loading: boolean) => {
        set((state: WorkspaceState) => ({
          isLoadingWorkspaceInfo: {
            ...state.isLoadingWorkspaceInfo,
            [workspace]: loading
          }
        }))
      }
    }),
    {
      name: 'workspace-storage',
      storage: createJSONStorage(() => localStorage),
      version: 1,
      // Only persist currentWorkspace, not the dynamic data
      partialize: (state: WorkspaceState) => ({
        currentWorkspace: state.currentWorkspace
      })
    }
  )
)

const useWorkspaceStore = createSelectors(useWorkspaceStoreBase)

export { useWorkspaceStore }


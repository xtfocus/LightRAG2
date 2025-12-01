import { useEffect, useState } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import Button from '@/components/ui/Button'
import { useWorkspaceStore } from '@/stores/workspace'
import { FolderIcon, RefreshCwIcon, InfoIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { errorMessage } from '@/lib/utils'

export default function WorkspaceSwitcher() {
  const { t } = useTranslation()
  const [isOpen, setIsOpen] = useState(false)
  
  const currentWorkspace = useWorkspaceStore.use.currentWorkspace()
  const workspaces = useWorkspaceStore.use.workspaces()
  const isLoadingWorkspaces = useWorkspaceStore.use.isLoadingWorkspaces()
  const workspaceInfo = useWorkspaceStore.use.workspaceInfo()
  const setCurrentWorkspace = useWorkspaceStore.use.setCurrentWorkspace()
  const refreshWorkspaces = useWorkspaceStore.use.refreshWorkspaces()
  const refreshWorkspaceInfo = useWorkspaceStore.use.refreshWorkspaceInfo()
  
  const isLoadingWorkspaceInfo = useWorkspaceStore.use.isLoadingWorkspaceInfo()
  const currentInfo = currentWorkspace ? workspaceInfo[currentWorkspace] : null
  const isLoadingInfo = currentWorkspace 
    ? isLoadingWorkspaceInfo[currentWorkspace] || false
    : false

  // Load workspaces on mount
  useEffect(() => {
    refreshWorkspaces().catch((error) => {
      console.error('Failed to load workspaces:', error)
      // Component will still render even if API fails
    })
  }, [refreshWorkspaces])

  // Refresh workspace info when workspace changes
  useEffect(() => {
    if (currentWorkspace) {
      refreshWorkspaceInfo(currentWorkspace)
    }
  }, [currentWorkspace, refreshWorkspaceInfo])

  const handleWorkspaceChange = (value: string) => {
    setCurrentWorkspace(value)
    setIsOpen(false)
  }

  const handleRefresh = async () => {
    try {
      await refreshWorkspaces()
      if (currentWorkspace) {
        await refreshWorkspaceInfo(currentWorkspace)
      }
    } catch (error) {
      console.error('Failed to refresh workspaces:', errorMessage(error))
    }
  }

  const displayValue = currentWorkspace || t('workspace.select', 'Select workspace')
  const selectValue = currentWorkspace || ''

  return (
    <div className="flex items-center gap-2" data-testid="workspace-switcher">
      <Select
        value={selectValue}
        onValueChange={handleWorkspaceChange}
        open={isOpen}
        onOpenChange={setIsOpen}
      >
        <SelectTrigger className="h-8 w-[180px] text-xs">
          <div className="flex items-center gap-2">
            <FolderIcon className="h-3.5 w-3.5" />
            <SelectValue placeholder={t('workspace.select', 'Select workspace')}>
              {displayValue}
            </SelectValue>
          </div>
        </SelectTrigger>
        <SelectContent>
          {isLoadingWorkspaces ? (
            <SelectItem value="__loading__" disabled>
              {t('workspace.loading', 'Loading...')}
            </SelectItem>
          ) : workspaces.length === 0 ? (
            <SelectItem value="__empty__" disabled>
              {t('workspace.noWorkspaces', 'No workspaces')}
            </SelectItem>
          ) : (
            workspaces.map((workspace) => (
              <SelectItem key={workspace} value={workspace}>
                {workspace}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>

      {currentWorkspace && (
        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              disabled={isLoadingInfo}
            >
              <InfoIcon className={cn(
                "h-4 w-4",
                isLoadingInfo && "animate-spin"
              )} />
            </Button>
          </PopoverTrigger>
          <PopoverContent side="bottom" align="end" className="w-64">
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold">{t('workspace.info', 'Workspace Info')}</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={handleRefresh}
                  disabled={isLoadingWorkspaces || isLoadingInfo}
                >
                  <RefreshCwIcon className={cn(
                    "h-3.5 w-3.5",
                    (isLoadingWorkspaces || isLoadingInfo) && "animate-spin"
                  )} />
                </Button>
              </div>
              
              {isLoadingInfo ? (
                <div className="text-xs text-muted-foreground">
                  {t('workspace.loadingInfo', 'Loading workspace information...')}
                </div>
              ) : currentInfo ? (
                <div className="flex flex-col gap-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('workspace.name', 'Workspace')}:</span>
                    <span className="font-medium">{currentInfo.workspace}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('workspace.documents', 'Documents')}:</span>
                    <span className="font-medium">{currentInfo.document_count}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('workspace.entities', 'Entities')}:</span>
                    <span className="font-medium">{currentInfo.entity_count}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('workspace.relations', 'Relations')}:</span>
                    <span className="font-medium">{currentInfo.relation_count}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('workspace.chunks', 'Chunks')}:</span>
                    <span className="font-medium">{currentInfo.chunk_count}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{t('workspace.pipeline', 'Pipeline')}:</span>
                    <span className={cn(
                      "font-medium",
                      currentInfo.pipeline_busy ? "text-amber-600 dark:text-amber-400" : "text-green-600 dark:text-green-400"
                    )}>
                      {currentInfo.pipeline_busy 
                        ? t('workspace.busy', 'Busy')
                        : t('workspace.idle', 'Idle')
                      }
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">
                  {t('workspace.noInfo', 'No workspace information available')}
                </div>
              )}
            </div>
          </PopoverContent>
        </Popover>
      )}

      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={handleRefresh}
        disabled={isLoadingWorkspaces}
        tooltip={t('workspace.refresh', 'Refresh workspaces')}
      >
        <RefreshCwIcon className={cn(
          "h-4 w-4",
          isLoadingWorkspaces && "animate-spin"
        )} />
      </Button>
    </div>
  )
}




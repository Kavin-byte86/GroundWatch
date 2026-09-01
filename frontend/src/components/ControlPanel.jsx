import { useState, useMemo } from 'react'

export default function ControlPanel({ nodes, onSend }) {
  const nodeIds = useMemo(() => Object.keys(nodes).sort(), [nodes])

  const [targetNode, setTargetNode] = useState('')
  const [tiltIntensity, setTiltIntensity] = useState(0.5)
  const [killSelection, setKillSelection] = useState({}) // { nodeId: boolean }

  const hasNodes = nodeIds.length > 0
  const effectiveTarget = targetNode || (hasNodes ? nodeIds[0] : '')

  const handleSimulateTilt = () => {
    if (!effectiveTarget) return
    onSend({ command: 'simulate_tilt', node_id: effectiveTarget, intensity: tiltIntensity })
  }

  const handleInjectBlast = () => {
    if (!effectiveTarget) return
    onSend({ command: 'inject_blast', node_id: effectiveTarget })
  }

  const handleKillNodes = () => {
    const selected = Object.entries(killSelection)
      .filter(([, v]) => v)
      .map(([k]) => k)
    if (selected.length === 0) return
    onSend({ command: 'kill_nodes', node_ids: selected })
  }

  const handleReset = () => {
    onSend({ command: 'reset' })
    setKillSelection({})
  }

  const toggleKill = (id) => {
    setKillSelection((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const killCount = Object.values(killSelection).filter(Boolean).length

  return (
    <div className="p-3 text-xs space-y-3">
      <div className="text-sm font-semibold text-panel-heading mb-2">Simulation Controls</div>

      {/* Target node picker */}
      <div>
        <label className="block text-panel-muted mb-1">Target Node</label>
        <select
          value={effectiveTarget}
          onChange={(e) => setTargetNode(e.target.value)}
          disabled={!hasNodes}
          className="w-full bg-panel-bg border border-panel-border rounded-sm px-2 py-1.5 text-panel-text focus:outline-none focus:border-panel-muted disabled:opacity-40"
        >
          {!hasNodes && <option value="">No nodes available</option>}
          {nodeIds.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </div>

      {/* Simulate tilt */}
      <div>
        <label className="block text-panel-muted mb-1">
          Tilt Intensity: {tiltIntensity.toFixed(2)}
        </label>
        <div className="flex items-center gap-2">
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={tiltIntensity}
            onChange={(e) => setTiltIntensity(parseFloat(e.target.value))}
            className="flex-1"
          />
          <button
            onClick={handleSimulateTilt}
            disabled={!effectiveTarget}
            className="px-3 py-1.5 border border-panel-border rounded-sm text-panel-text hover:bg-panel-border disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Simulate Tilt Increase
          </button>
        </div>
      </div>

      {/* Inject blast */}
      <div>
        <button
          onClick={handleInjectBlast}
          disabled={!effectiveTarget}
          className="px-3 py-1.5 border border-panel-border rounded-sm text-panel-text hover:bg-panel-border disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Inject Blast Vibration
        </button>
      </div>

      {/* Kill nodes */}
      <div>
        <label className="block text-panel-muted mb-1">Kill Nodes</label>
        {!hasNodes ? (
          <div className="text-panel-muted">No nodes available</div>
        ) : (
          <div className="flex flex-wrap gap-1.5 mb-2 max-h-20 overflow-y-auto">
            {nodeIds.map((id) => (
              <label
                key={id}
                className={`flex items-center gap-1 px-2 py-1 border rounded-sm cursor-pointer select-none ${
                  killSelection[id]
                    ? 'border-risk-warning text-risk-warning'
                    : 'border-panel-border text-panel-muted hover:text-panel-text'
                }`}
              >
                <input
                  type="checkbox"
                  checked={!!killSelection[id]}
                  onChange={() => toggleKill(id)}
                  className="sr-only"
                />
                {id}
              </label>
            ))}
          </div>
        )}
        <button
          onClick={handleKillNodes}
          disabled={killCount === 0}
          className="px-3 py-1.5 border border-panel-border rounded-sm text-panel-text hover:bg-panel-border disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Kill Selected Nodes ({killCount})
        </button>
      </div>

      {/* Reset */}
      <div className="pt-1 border-t border-panel-border">
        <button
          onClick={handleReset}
          className="px-3 py-1.5 border border-panel-border rounded-sm text-panel-text hover:bg-panel-border"
        >
          Reset Simulation
        </button>
      </div>
    </div>
  )
}

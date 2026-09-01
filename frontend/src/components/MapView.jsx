import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Polygon, CircleMarker, Popup, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

// Placeholder mine panel boundary (Jharia coalfield area)
const PANEL_CENTER = [23.7644, 86.4131]
const PANEL_BOUNDARY = [
  [23.7660, 86.4110],
  [23.7660, 86.4152],
  [23.7628, 86.4152],
  [23.7628, 86.4110],
]

const RISK_COLORS = {
  safe: '#22c55e',
  watch: '#eab308',
  warning: '#f97316',
  critical: '#ef4444',
}

function getMarkerColor(node) {
  if (node.status === 'excluded' || node.status === 'silent') {
    return '#6b7280'
  }
  return RISK_COLORS[node.risk_tier] || '#6b7280'
}

function getMarkerOptions(node, isSelected) {
  const isInactive = node.status === 'excluded' || node.status === 'silent'
  return {
    radius: 8,
    fillColor: getMarkerColor(node),
    fillOpacity: isInactive ? 0.4 : 0.85,
    color: isSelected ? '#f9fafb' : isInactive ? '#6b7280' : getMarkerColor(node),
    weight: isSelected ? 2.5 : 1.5,
    dashArray: isInactive ? '4 3' : null,
  }
}

// Sub-component to handle map view fitting
function MapController() {
  const map = useMap()
  useEffect(() => {
    map.setView(PANEL_CENTER, 16)
  }, [map])
  return null
}

export default function MapView({ nodes, selectedNodeId, onSelectNode }) {
  const nodeList = Object.values(nodes)

  return (
    <div className="w-full h-full">
      <MapContainer
        center={PANEL_CENTER}
        zoom={16}
        className="w-full h-full"
        zoomControl={true}
        attributionControl={false}
      >
        <MapController />
        <TileLayer
          url="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png"
          maxZoom={19}
        />

        {/* Underground panel boundary */}
        <Polygon
          positions={PANEL_BOUNDARY}
          pathOptions={{
            color: '#6b7280',
            weight: 1.5,
            fillColor: '#2a2e3a',
            fillOpacity: 0.3,
            dashArray: '6 4',
          }}
        />

        {/* Node markers */}
        {nodeList.map((node) => (
          <CircleMarker
            key={node.node_id}
            center={[node.lat, node.lng]}
            {...getMarkerOptions(node, node.node_id === selectedNodeId)}
            eventHandlers={{
              click: () => onSelectNode(node.node_id),
            }}
          >
            <Popup>
              <div className="text-xs leading-snug" style={{ color: '#181b24' }}>
                <div className="font-semibold mb-1">{node.node_id}</div>
                <div>Status: {node.status}</div>
                <div>Risk: {node.risk_tier} ({node.risk_score})</div>
                <div>Tilt: {node.tilt_deg}°</div>
                <div>Vibration: {node.vibration_rms} g</div>
                <div>Strain: {node.strain_mm} mm</div>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  )
}

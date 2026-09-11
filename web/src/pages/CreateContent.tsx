import React, { useState } from 'react'

export default function CreateContent(){
  const [prompt, setPrompt] = useState('')
  const [title, setTitle] = useState('')
  const [duration, setDuration] = useState(60)

  async function handleGenerate(){
    const payload = { title: title || 'Untitled', prompt, content_type: 'DSA', duration_seconds: duration }
    const res = await fetch('/api/projects', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    })
    const data = await res.json()
    alert('Project created: ' + data.id)
  }

  return (
    <div style={{maxWidth: 900}}>
      <label>Title</label>
      <input style={{width: '100%'}} value={title} onChange={e => setTitle(e.target.value)} />

      <label style={{marginTop: 12}}>Prompt</label>
      <textarea style={{width: '100%', height: 160}} value={prompt} onChange={e => setPrompt(e.target.value)} />

      <label style={{marginTop: 12}}>Duration (seconds)</label>
      <input type="number" value={duration} onChange={e => setDuration(Number(e.target.value))} />

      <div style={{marginTop: 16}}>
        <button onClick={handleGenerate}>Create Project</button>
      </div>
    </div>
  )
}

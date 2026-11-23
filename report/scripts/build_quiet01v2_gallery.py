"""Local video gallery, explicit pending state and recorded replay outcomes."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'report'


def main():
    cards=[];rows=[]
    for model in ('xshape','v2','v1'):
        meta_path=REPORT/f'data/quiet01v2/videos/{model}/replay.json'
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else None
        video=REPORT/f'videos/quiet01v2_{model}_best.mp4'
        title={'xshape':'X-shape','v2':'V2','v1':'V1'}[model]
        content=f'<h2>{title}: saved best policy</h2>'
        if video.exists():
            content+=f'<video controls preload="metadata" poster="figures/quiet01v2_{model}_preview.jpg"><source src="videos/{video.name}" type="video/mp4"></video><p><a href="videos/{video.name}">Open or download MP4</a></p>'
        else:content+='<p class="pending">Video preparation is pending. Reload this page after completion.</p>'
        if meta:
            content+=f'<p>Checkpoint: +{meta["checkpoint_step"]/1e6:.2f}M training transitions. Replay backend: MJX {meta.get("backend","cuda")}. Selected replay: {meta["replay_selected_passes"]}/4; archived same four trials: {sum(meta["archived_selected_passed"])}/4. Full archived GPU evaluation: {meta["archived_retention_passes"]}/96. Selected verdict parity: {meta["parity"]}.</p>'
            content+='<table><tr><th>Panel</th><th>Actual initial motor RMS</th><th>Initial five-second hold</th><th>20-second retention</th></tr>'
            for i,trial in enumerate(meta['selected_trials']):
                verdict='PASS' if meta['per_trial']['passed'][trial] else 'FAIL'
                start=meta['per_trial']['settling_s'][trial]
                initial_hold=f'PASS (completed about {start+5:.2f}s)' if start>=0 else 'Not completed'
                content+=f'<tr><td>{html.escape(meta["selected_labels"][i])}</td><td>{meta["initial_motor_rms_deg"][i]:.2f} deg</td><td>{initial_hold}</td><td>{verdict}, trial {trial}</td></tr>'
            content+='</table>'
            rows.append(f'| {title} | {meta["checkpoint_step"]/1e6:.2f} | {meta.get("backend","cuda")} | {meta["replay_selected_passes"]}/4 | {meta["archived_retention_passes"]}/96 | '+('Ready' if video.exists() else 'Rendering')+' |')
        else:rows.append(f'| {title} | pending | pending | pending | pending | Waiting for replay |')
        cards.append('<section>'+content+'</section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QUIET-01 v2 policy videos</title><style>body{font:17px system-ui,sans-serif;margin:auto;padding:24px;max-width:1320px;background:#111925;color:#e7edf6}a{color:#86c9ff}section{background:#1c2738;border-radius:12px;padding:20px;margin:24px 0}video{width:100%;max-height:85vh;background:black}p{line-height:1.5}table{border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #465467}button{padding:8px 18px;margin:5px;cursor:pointer}.pending{color:#ffd28c}</style>
<h1>Quiet-pose recovery: inspect the best policies</h1><p>Each 20-second video compares four fixed test cases: quiet reference (top left), moderate deviation (top right), larger deviation (bottom left), hard deviation (bottom right). There are no external pushes. Left foot is blue; right foot is orange.</p>
<p>Angles label the requested reset noise before physical projection. Each is a different random pose and direction, not a scaled version of another panel. A nominally larger sample can be easier for the policy. Actual initial motor deviations are shown.</p>
<p>The video PASS/FAIL label is the full 20-second retention outcome. A robot can complete its initial five-second hold and still fail if it drifts later; the table separates these two results. A terminated episode freezes its last valid pose with a label; no restart is spliced into the clip.</p>
<p>Playback speed: <button onclick="speed(.5)">0.5x</button><button onclick="speed(1)">1x</button><button onclick="speed(2)">2x</button></p>
<p><a href="quiet01-v2-results.md">Results and learning curves</a> · <a href="quiet01-v2-assessment.md">Assessment</a></p>'''+''.join(cards)+'''<script>function speed(rate){document.querySelectorAll('video').forEach(v=>v.playbackRate=rate)}</script></html>'''
    (REPORT/'quiet01v2-videos.html').write_text(page,encoding='utf-8')
    media=['# QUIET-01 v2 best-policy videos','',
        '[Open the video gallery](quiet01v2-videos.html).','',
        'Requested23November2026. Each20s clip shows four fixed trials side by side:',
        'quiet0, moderate48, larger64, hard80. Trials are fixed before replay outcomes.',
        'Nominal requested initial noise0/3/5/8deg is projected onto physically valid poses;',
        'actual initial motor RMS is shown. Full sensor noise, no pushes, seed1234,',
        'original96-member per-trial PRNG keys. Only the four displayed independent trials',
        'are replayed. Each model uses its selected best RETURN',
        'checkpoint and the stricter20s retention evaluation.','',
        '| Model | Checkpoint additional M | Replay backend | Replayed retention | Archived GPU retention | Video |',
        '|---|---:|---|---:|---:|---|',*rows,'',
        'GPU replay3263991 was cancelled while still pending after long queue estimates.',
        'The full-batch CPU attempt3263994 was stopped to reduce replay work.',
        'CPU-only selected-trial MJX replay3264005 uses the same frozen source/weights/model/solver and',
        'float32 arithmetic. Backend numerical differences may change trajectories and',
        'verdicts; replay outcomes are reported separately from archived GPU scores.',
        'There are no learning updates and no changes to reward, targets or physics.',
        'Display-only lighting and foot colors improve visibility.','',
        'Videos run at1x,20fps,1280x960,H264. The gallery offers0.5x/1x/2x controls.',
        'Failure freezes the last valid pose and labels the termination; there is no',
        'reset splice or physics resimulation during rendering. Blue=leftfoot,orange=rightfoot.',
        'Foot-position error is relative to the torso; it is not clearance.','',
        'Raw traces, replay metadata and video provenance are in',
        '`report/data/quiet01v2/videos/{model}/`; pipeline state in',
        '`report/data/quiet01v2/videos/pipeline.json`.','']
    media+=['## Clarification of the visible outcomes','',
        'X-shape trial48 completes the initial five-second hold (start0.59s) and',
        'has a longest continuous quiet streak12.05s, but fails the final retention',
        'window. Terminal sole-relative error20.56mm exceeds20mm, by0.56mm;',
        'peak during the final five seconds21.35mm. No fall or task termination.',
        'Thus the clip shows successful initial recovery with marginal later drift.','',
        'V1 trial48 ends at1.54s on the planted-foot rule: native FK of recorded',
        'states gives left sole-center drift54.62mm versus50mm allowed, right8.45mm.',
        'No whole-foot air episode above5mm and no physical fall. Trial64 succeeds:',
        'peak sole-center drifts2.74/10.73mm; final quiet streak10.25s.','',
        'The two V1 starts are different joint patterns: initial motor RMS2.19/2.58deg;',
        'left-thigh offsets+2.94/-3.36deg respectively. Their noise-bank labels',
        'do not order the difficulty of these particular samples.','']
    (REPORT/'quiet01v2-media.md').write_text('\n'.join(media),encoding='utf-8')
    print('Gallery updated',flush=True)


if __name__=='__main__':main()

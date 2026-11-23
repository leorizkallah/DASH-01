import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'report';BASE=REPORT/'data/quiet_push01'


def main():
    cards=[]
    for model in ('xshape','v2','v1'):
        folder=BASE/model;summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else None
        content='<h2>'+{'xshape':'X-shape','v2':'V2','v1':'V1'}[model]+'</h2>'
        if summary:
            controls=summary['controls']
            content+=f'<p>Best checkpoint +{summary["checkpoint_step"]/1e6:.2f}M. Zero-force controls: {sum(c["survived"] for c in controls)}/2 survive, {sum(c["retained"] for c in controls)}/2 finish with a quiet hold. Weights unchanged: {summary["no_learning"]}.</p>'
        for direction in ('forward','left'):
            name=f'quiet_push01_{model}_{direction}.mp4';p=REPORT/'videos'/name
            content+='<h3>'+direction.capitalize()+' push</h3>'
            if p.exists() and (folder/f'video-{direction}-provenance.json').exists():
                content+=f'<video controls preload="metadata" poster="figures/quiet_push01_{model}_{direction}_preview.jpg"><source src="videos/{name}" type="video/mp4"></video><p><a href="videos/{name}">Open or download video</a></p>'
            else:content+='<p class="pending">Preparing simulation/video; reload after completion.</p>'
        if summary:content+=f'<p><a href="data/quiet_push01/{model}/trials.csv">All trial measurements(CSV)</a></p>'
        cards.append('<section>'+content+'</section>')
    graph='<img src="figures/quiet_push01_outcomes.png" alt="Push outcome counts by force and direction">' if (REPORT/'figures/quiet_push01_outcomes.png').exists() else ''
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Push tests: best quiet policies</title>
<style>body{font:17px system-ui,sans-serif;margin:auto;padding:24px;max-width:1320px;background:#111925;color:#e7edf6}section{background:#1c2738;padding:20px;margin:24px 0;border-radius:12px}p{line-height:1.5}video,img{width:100%;max-height:90vh;object-fit:contain}a{color:#86c9ff}button{padding:8px 18px;margin:4px}.pending{color:#ffd28c}</style>
<h1>Push simulations: fixed best policies, no training</h1><p>Each video shows four independent fresh-start trials: 0,5,20 and60N. A single horizontal100ms pulse acts at the torso centre of mass at6s. Red arrows show the force while active. Blue left foot, orange right foot.</p>
<p>All models use their saved best quiet-pose policy, unchanged. Foot lifts and shifts are allowed during this evaluation so recovery remains visible. Survival, regaining a five-second quiet hold, and a quiet final five-second window are reported separately. Physical failures freeze visibly; no reset is spliced into a video.</p>
<p>The plots cover four directions and two paired noise realizations,66trials per model. Videos show the first noise realization, forward and left directions. Counts out of two are exploratory observations, not proven safety limits.</p>
<p>Speed: <button onclick="rate(.25)">0.25x</button><button onclick="rate(.5)">0.5x</button><button onclick="rate(1)">1x</button></p>
<p><a href="quiet-push01-results.md">Protocol, plots and results</a> · <a href="quiet01v2-videos.html">Original quiet-pose videos</a></p>'''+graph+''.join(cards)+'''<script>function rate(r){document.querySelectorAll('video').forEach(v=>v.playbackRate=r)}</script></html>'''
    (REPORT/'quiet-push01-videos.html').write_text(page,encoding='utf-8')
    print('Push gallery updated',flush=True)


if __name__=='__main__':main()

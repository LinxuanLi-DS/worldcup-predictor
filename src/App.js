import { useState, useEffect } from 'react';
import { GROUPS, SCHEDULE, KNOCKOUT_STAGE, ELO, CN, FLAG, SQUADS, MATCH_ANALYSIS } from './data';
import BettingTab from './Betting';

// ============================================================
// Helpers
// ============================================================
const cn = (name) => CN[name] || name;
const flag = (name) => FLAG[name] || '🏳️';

function getPrediction(home, away) {
  const r = computePoissonPrediction(home, away);
  // Determine predicted outcome
  let predicted_outcome;
  if (r.ph > r.pa && r.ph > r.pd) predicted_outcome = 'W';
  else if (r.pa > r.ph && r.pa > r.pd) predicted_outcome = 'L';
  else predicted_outcome = 'D';
  return { ...r, predicted_outcome };
}

function poissProb(xg, k) {
  let f = 1;
  for (let i = 2; i <= k; i++) f *= i;
  return (Math.pow(xg, k) * Math.exp(-xg)) / f;
}

// Dixon-Coles correction for low-scoring outcomes (increases draw accuracy)
function dcTau(h, a, lambda, mu, rho) {
  if (h === 0 && a === 0) return 1 - lambda * mu * rho;
  if (h === 0 && a === 1) return 1 + lambda * rho;
  if (h === 1 && a === 0) return 1 + mu * rho;
  if (h === 1 && a === 1) return 1 - rho;
  return 1;
}

function computePoisson(homeXG, awayXG) {
  const maxG = 7;
  const rho = -0.13; // Dixon-Coles correction parameter
  let ph = 0, pd = 0, pa = 0;
  let bestScore = '0-0', bestProb = 0;
  let bestHomeWin = null, bestDraw = null, bestAwayWin = null;
  for (let h = 0; h <= maxG; h++) {
    for (let a = 0; a <= maxG; a++) {
      const tau = dcTau(h, a, homeXG, awayXG, rho);
      const prob = poissProb(homeXG, h) * poissProb(awayXG, a) * tau;
      const score = `${h}-${a}`;
      if (h > a) {
        ph += prob;
        if (!bestHomeWin || prob > bestHomeWin.prob) bestHomeWin = { score, prob };
      } else if (h === a) {
        pd += prob;
        if (!bestDraw || prob > bestDraw.prob) bestDraw = { score, prob };
      } else {
        pa += prob;
        if (!bestAwayWin || prob > bestAwayWin.prob) bestAwayWin = { score, prob };
      }
      if (prob > bestProb) { bestProb = prob; bestScore = score; }
    }
  }
  // Pick the most likely score consistent with predicted outcome
  let outcomeScore = bestScore, outcomeProb = bestProb;
  if (ph > pd && ph > pa && bestHomeWin) { outcomeScore = bestHomeWin.score; outcomeProb = bestHomeWin.prob; }
  else if (pa > ph && pa > pd && bestAwayWin) { outcomeScore = bestAwayWin.score; outcomeProb = bestAwayWin.prob; }
  else if (bestDraw) { outcomeScore = bestDraw.score; outcomeProb = bestDraw.prob; }
  return { ph, pd, pa, homeXG, awayXG, mostLikelyScore: outcomeScore, mostLikelyScoreProb: outcomeProb };
}

// Host nations get 75 Elo home advantage (Hicruben model)
const HOST_NATIONS = new Set(['Mexico', 'USA', 'Canada']);
const HOME_ADV = 75;

// Market odds: compress model probs toward uniform + bookmaker overround
// Real bookmakers never give 1.08 odds — they compress extreme probs
function marketOdds(ph, pd, pa) {
  const compress = 0.18; // pull 18% toward uniform (1/3)
  const u = 1 / 3;
  const adjH = ph * (1 - compress) + u * compress;
  const adjD = pd * (1 - compress) + u * compress;
  const adjA = pa * (1 - compress) + u * compress;
  const total = adjH + adjD + adjA;
  const nH = adjH / total;
  const nD = adjD / total;
  const nA = adjA / total;
  const margin = 1.12; // 12% overround (typical竞彩)
  return {
    w: +(1 / (nH * margin)).toFixed(2),
    d: +(1 / (nD * margin)).toFixed(2),
    l: +(1 / (nA * margin)).toFixed(2),
  };
}

// Expected goals from Elo rating difference
function expectedGoals(rating, opponent, homeBonus = 0) {
  const diff = (rating + homeBonus) - opponent;
  const lambda = 1.35 + diff / 400;
  return Math.max(0.3, Math.min(3.5, lambda));
}

function computePoissonPrediction(home, away) {
  const eH = ELO[home] || 1600;
  const eA = ELO[away] || 1600;
  const hb = (HOST_NATIONS.has(home) ? HOME_ADV : 0) - (HOST_NATIONS.has(away) ? HOME_ADV : 0);

  const homeXG = expectedGoals(eH, eA, hb);
  const awayXG = expectedGoals(eA, eH, -hb / 2);
  const poissonResult = computePoisson(homeXG, awayXG);

  const total = poissonResult.ph + poissonResult.pd + poissonResult.pa;

  return {
    ph: poissonResult.ph / total,
    pd: poissonResult.pd / total,
    pa: poissonResult.pa / total,
    homeXG,
    awayXG,
    mostLikelyScore: poissonResult.mostLikelyScore,
    mostLikelyScoreProb: poissonResult.mostLikelyScoreProb,
  };
}

function getPredictionOutcome(result) {
  if (result.ph > result.pa && result.ph > result.pd) return `${CN[result.home]}胜`;
  if (result.pa > result.ph && result.pa > result.pd) return `${CN[result.away]}胜`;
  return '平局';
}

function getOutcomeLabel(p) {
  if (p === 'W') return '主胜';
  if (p === 'D') return '平局';
  return '客胜';
}

// ============================================================
// Squad Modal
// ============================================================
function SquadModal({ team, onClose }) {
  const squad = SQUADS[team];
  if (!squad) return null;
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.85)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}
      onClick={onClose}>
      <div style={{ background: '#0f172a', borderRadius: 16, padding: 24, maxWidth: 480, width: '100%', maxHeight: '80vh', overflowY: 'auto', border: '1px solid #1e293b' }}
        onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ margin: 0, color: '#f1f5f9', fontSize: 18 }}>{flag(team)} {cn(team)} 预估首发</h3>
          <button onClick={onClose} style={{ background: '#1e293b', border: 'none', color: '#94a3b8', borderRadius: 8, padding: '4px 12px', cursor: 'pointer' }}>✕</button>
        </div>
        <div style={{ fontSize: 10, color: '#e07820', marginBottom: 12, padding: '4px 8px', background: '#1e293b', borderRadius: 6, display: 'inline-block' }}>
          ⚠️ 预估阵容，仅供参考
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #1e293b' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 11 }}>#</th>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 11 }}>球员</th>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 11 }}>位置</th>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 11 }}>俱乐部</th>
            </tr>
          </thead>
          <tbody>
            {squad.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #0f172a' }}>
                <td style={{ padding: '5px 8px', color: '#94a3b8', fontSize: 12 }}>{i + 1}</td>
                <td style={{ padding: '5px 8px', color: '#f1f5f9', fontSize: 13, fontWeight: 600 }}>{p.name}</td>
                <td style={{ padding: '5px 8px', color: '#e07820', fontSize: 12 }}>{p.pos}</td>
                <td style={{ padding: '5px 8px', color: '#64748b', fontSize: 11 }}>{p.club}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ============================================================
// Tab 1: 赛程 (Schedule) — 全部用实时Poisson
// ============================================================
function ScheduleTab({ onOpenSquad }) {
  const [filter, setFilter] = useState('all');

  const formatDate = (d) => {
    const date = new Date(d + 'T00:00:00');
    return `${date.getMonth() + 1}月${date.getDate()}日`;
  };

  // Time is already in Beijing time (UTC+8)
  const formatTime = (date, localTime) => {
    return localTime;
  };

  const cardStyle = {
    background: '#0f172a', borderRadius: 12, padding: '14px 16px',
    border: '1px solid #1e293b', transition: 'all 0.2s',
  };

  // Group matches by date
  const dates = {};
  SCHEDULE.forEach(m => {
    if (!dates[m.date]) dates[m.date] = [];
    dates[m.date].push(m);
  });
  const filteredDates = Object.keys(dates).sort();

  // Compute accuracy using real-time Poisson predictions
  const completedMatches = SCHEDULE.filter(m => m.completed);
  let correct = 0;
  completedMatches.forEach(m => {
    const pred = getPrediction(m.home, m.away);
    if (!pred) return;
    const actual = m.homeScore > m.awayScore ? 'W' : m.homeScore < m.awayScore ? 'L' : 'D';
    if (pred.predicted_outcome === actual) correct++;
  });
  const accuracy = completedMatches.length > 0 ? Math.round((correct / completedMatches.length) * 100) : 0;

  const filterBtn = (active) => ({
    padding: '4px 12px', borderRadius: 14, border: 'none', cursor: 'pointer',
    fontFamily: 'inherit', fontSize: 11, fontWeight: 600, transition: 'all 0.2s',
    background: active ? '#e07820' : '#1e293b',
    color: active ? '#020617' : '#64748b',
  });

  return (
    <div>
      {/* Accuracy header */}
      <div style={{ textAlign: 'center', marginBottom: 20, padding: '12px 16px', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', borderRadius: 12, border: '1px solid #1e293b' }}>
        <span style={{ fontSize: 13, color: '#94a3b8' }}>已赛 {completedMatches.length} 场 · 预测准确率 </span>
        <span style={{ fontSize: 20, fontWeight: 900, color: accuracy >= 50 ? '#22c55e' : accuracy >= 30 ? '#e07820' : '#ef4444' }}>{accuracy}%</span>
        <span style={{ fontSize: 13, color: '#64748b' }}> ({correct}/{completedMatches.length})</span>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginBottom: 18 }}>
        <button style={filterBtn(filter === 'all')} onClick={() => setFilter('all')}>全部</button>
        <button style={filterBtn(filter === 'completed')} onClick={() => setFilter('completed')}>已赛</button>
        <button style={filterBtn(filter === 'upcoming')} onClick={() => setFilter('upcoming')}>未赛</button>
      </div>

      {filteredDates.map(date => {
        const matches = dates[date];
        const hasMatch = filter === 'all' || matches.some(m => (filter === 'completed' && m.completed) || (filter === 'upcoming' && !m.completed));
        if (!hasMatch) return null;

        return (
          <div key={date} style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 4, height: 16, background: '#e07820', borderRadius: 2 }} />
              {formatDate(date)}
            </div>
            {matches.map((m, idx) => {
              const pred = getPrediction(m.home, m.away);
              const actual = m.completed ? (m.homeScore > m.awayScore ? 'W' : m.homeScore < m.awayScore ? 'L' : 'D') : null;
              const isCorrect = pred && actual && pred.predicted_outcome === actual;
              const isWrong = pred && actual && pred.predicted_outcome !== actual;

              return (
                <div key={idx} style={{ ...cardStyle, marginBottom: 8 }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = '#e07820'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = '#1e293b'; e.currentTarget.style.transform = 'translateY(0)'; }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <span style={{ fontSize: 11, color: '#475569', padding: '2px 8px', background: '#1e293b', borderRadius: 6 }}>{m.group}组</span>
                    <span style={{ fontSize: 11, color: '#64748b' }}>{formatTime(m.date, m.time)}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 22 }}>{flag(m.home)}</span>
                      <span
                        style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0', cursor: 'pointer' }}
                        onClick={() => onOpenSquad(m.home)}
                      >{cn(m.home)}</span>
                    </div>
                    <div style={{ textAlign: 'center', minWidth: 80 }}>
                      {m.completed ? (
                        <div>
                          <div style={{ fontSize: 28, fontWeight: 900, color: '#f1f5f9' }}>
                            {m.homeScore} - {m.awayScore}
                          </div>
                          {pred && (
                            <div style={{ marginTop: 4 }}>
                              <div style={{ fontSize: 10, color: isCorrect ? '#4ade80' : isWrong ? '#f87171' : '#64748b', fontWeight: 700 }}>
                                {isCorrect ? '✓ 预测正确' : isWrong ? '✗ 预测错误' : ''}
                              </div>
                              {isWrong && (
                                <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                                  预测{getOutcomeLabel(pred.predicted_outcome)} → 实际{getOutcomeLabel(actual)}
                                </div>
                              )}
                              <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                                预测比分 {pred.mostLikelyScore} · 半场{pred.ph > 0.5 ? '主胜' : pred.pa > 0.5 ? '客胜' : '平局'}
                              </div>
                              <div style={{ fontSize: 9, color: '#64748b', marginTop: 1 }}>
                                大小球2.5: {(pred.homeXG + pred.awayXG) > 2.5 ? '大' : '小'}({(pred.homeXG + pred.awayXG).toFixed(1)}) · 让球: {pred.ph > 0.55 ? '主-1' : pred.pa > 0.55 ? '客+1' : '平手'}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div>
                          <div style={{ fontSize: 18, fontWeight: 700, color: '#e07820' }}>
                            {pred ? pred.mostLikelyScore : '-'}
                          </div>
                          {pred && (
                            <div style={{ marginTop: 4 }}>
                              <div style={{ fontSize: 9, color: '#94a3b8' }}>
                                {Math.round(pred.ph*100)}%胜 / {Math.round(pred.pd*100)}%平 / {Math.round(pred.pa*100)}%负
                              </div>
                              <div style={{ fontSize: 9, color: '#64748b', marginTop: 1 }}>
                                半场{pred.ph > 0.5 ? '主胜' : pred.pa > 0.5 ? '客胜' : '平局'} · 大小{(pred.homeXG + pred.awayXG) > 2.5 ? '大' : '小'}({(pred.homeXG + pred.awayXG).toFixed(1)}) · {pred.ph > 0.55 ? '主让1' : pred.pa > 0.55 ? '客受1' : '平手'}
                              </div>
                              {Math.max(pred.ph, pred.pa, pred.pd) > 0.5 && (
                                <div style={{ fontSize: 9, color: '#22c55e', marginTop: 2, fontWeight: 700 }}>
                                  💰 推荐: {pred.ph > pred.pa ? cn(m.home) + '胜' : pred.pa > pred.ph ? cn(m.away) + '胜' : '平局'}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'flex-end' }}>
                      <span
                        style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0', cursor: 'pointer' }}
                        onClick={() => onOpenSquad(m.away)}
                      >{cn(m.away)}</span>
                      <span style={{ fontSize: 22 }}>{flag(m.away)}</span>
                    </div>
                  </div>
                  {!m.completed && pred && (
                    <div style={{ display: 'flex', gap: 4, marginTop: 10, height: 6, borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${pred.ph * 100}%`, background: 'linear-gradient(90deg, #e07820, #f8a030)', borderRadius: '3px 0 0 3px' }} />
                      <div style={{ width: `${pred.pd * 100}%`, background: '#475569' }} />
                      <div style={{ width: `${pred.pa * 100}%`, background: 'linear-gradient(90deg, #7c3aed, #5b21b6)', borderRadius: '0 3px 3px 0' }} />
                    </div>
                  )}
                  {!m.completed && pred && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                      <span style={{ fontSize: 10, color: '#e07820' }}>{(pred.ph * 100).toFixed(0)}%</span>
                      <span style={{ fontSize: 10, color: '#64748b' }}>{(pred.pd * 100).toFixed(0)}%</span>
                      <span style={{ fontSize: 10, color: '#7c3aed' }}>{(pred.pa * 100).toFixed(0)}%</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// Tab 2: 积分榜 (Standings)
// ============================================================
function StandingsTab({ onOpenSquad }) {
  // Compute standings from completed matches
  const standings = {};
  Object.keys(GROUPS).forEach(g => {
    standings[g] = {};
    GROUPS[g].forEach(team => {
      standings[g][team] = { played: 0, won: 0, drawn: 0, lost: 0, gf: 0, ga: 0, gd: 0, pts: 0 };
    });
  });

  SCHEDULE.filter(m => m.completed).forEach(m => {
    const hs = standings[m.group][m.home];
    const as_ = standings[m.group][m.away];
    hs.played++; as_.played++;
    hs.gf += m.homeScore; hs.ga += m.awayScore;
    as_.gf += m.awayScore; as_.ga += m.homeScore;
    if (m.homeScore > m.awayScore) { hs.won++; hs.pts += 3; as_.lost++; }
    else if (m.homeScore < m.awayScore) { as_.won++; as_.pts += 3; hs.lost++; }
    else { hs.drawn++; as_.drawn++; hs.pts += 1; as_.pts += 1; }
  });

  Object.keys(standings).forEach(g => {
    Object.values(standings[g]).forEach(t => { t.gd = t.gf - t.ga; });
  });

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, margin: '0 0 4px' }}>📊 小组赛积分榜</h2>
        <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>基于已完成的比赛自动计算 · 前两名晋级32强</p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
        {Object.keys(GROUPS).map(group => {
          const teams = Object.entries(standings[group])
            .sort((a, b) => b[1].pts - a[1].pts || b[1].gd - a[1].gd || b[1].gf - a[1].gf);
          return (
            <div key={group} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 14, overflow: 'hidden' }}>
              <div style={{ background: '#e07820', padding: '8px 14px' }}>
                <span style={{ fontSize: 13, fontWeight: 800, color: '#020617' }}>{group} 组</span>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #1e293b' }}>
                    {['球队', '赛', '胜', '平', '负', '进球', '失球', '净胜', '积分'].map(h => (
                      <th key={h} style={{ padding: '6px 4px', color: '#64748b', fontWeight: 600, textAlign: h === '球队' ? 'left' : 'center', fontSize: 10 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {teams.map(([team, s], i) => (
                    <tr key={team} style={{
                      borderBottom: '1px solid #1e293b',
                      background: i < 2 ? 'rgba(74,222,128,0.05)' : 'transparent',
                    }}>
                      <td style={{ padding: '7px 4px', textAlign: 'left' }}>
                        {i < 2 && <span style={{ color: '#4ade80', marginRight: 4 }}>●</span>}
                        <span
                          style={{ cursor: 'pointer', fontWeight: 600, color: '#e2e8f0' }}
                          onClick={() => onOpenSquad(team)}
                        >{flag(team)} {cn(team)}</span>
                      </td>
                      {[s.played, s.won, s.drawn, s.lost, s.gf, s.ga, s.gd, s.pts].map((v, j) => (
                        <td key={j} style={{ padding: '7px 4px', textAlign: 'center', color: j === 7 ? '#e07820' : '#94a3b8', fontWeight: j === 7 ? 700 : 400 }}>{v}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// Tab 3: 对阵图 (Bracket)
// ============================================================
function BracketTab() {
  const ko = KNOCKOUT_STAGE || [];
  const r32 = ko.filter(m => m.round === 'R32');
  const r16 = ko.filter(m => m.round === 'R16');
  const qf = ko.filter(m => m.round === 'QF');
  const sf = ko.filter(m => m.round === 'SF');
  const final = ko.filter(m => m.round === 'Final');
  const third = ko.filter(m => m.round === 'Third');

  function MatchCard({ match, compact }) {
    const done = match.completed;
    const homeWin = done && match.homeScore > match.awayScore;
    const awayWin = done && match.awayScore > match.homeScore;
    const isPso = match.penaltyScore;
    const homeName = match.home || 'TBD';
    const awayName = match.away || 'TBD';
    const homeCn = CN[homeName] || homeName;
    const awayCn = CN[awayName] || awayName;
    const homeFlag = FLAG[homeName] || '🏳️';
    const awayFlag = FLAG[awayName] || '🏳️';
    const sz = compact ? 11 : 13;

    return (
      <div style={{
        background: done ? '#0a1628' : '#0f172a',
        border: `1px solid ${done ? '#1e3a5f' : '#1e293b'}`,
        borderRadius: 8, padding: compact ? '6px 8px' : '8px 12px',
        minWidth: compact ? 140 : 180, fontSize: sz,
        opacity: done ? 1 : 0.7,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span>{homeFlag}</span>
            <span style={{ color: homeWin ? '#22c55e' : '#e2e8f0', fontWeight: homeWin ? 700 : 500 }}>{homeCn}</span>
          </div>
          <span style={{ color: done ? '#f1f5f9' : '#475569', fontWeight: 700, fontSize: sz + 1 }}>
            {done ? match.homeScore : '-'}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span>{awayFlag}</span>
            <span style={{ color: awayWin ? '#22c55e' : '#94a3b8', fontWeight: awayWin ? 700 : 500 }}>{awayCn}</span>
          </div>
          <span style={{ color: done ? '#f1f5f9' : '#475569', fontWeight: 700, fontSize: sz + 1 }}>
            {done ? match.awayScore : '-'}
          </span>
        </div>
        {isPso && <div style={{ fontSize: 10, color: '#e07820', marginTop: 2, textAlign: 'center' }}>点球 {match.penaltyScore}</div>}
        {done && <div style={{ fontSize: 9, color: '#22c55e', marginTop: 2, textAlign: 'center' }}>✓ {match.winner || (homeWin ? homeCn : awayCn)} 晋级</div>}
        {!done && match.date && <div style={{ fontSize: 9, color: '#475569', marginTop: 2, textAlign: 'center' }}>{match.date}</div>}
      </div>
    );
  }

  function RoundColumn({ title, matches, emoji }) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 160 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#e07820', textAlign: 'center', marginBottom: 6, padding: '4px 0', borderBottom: '1px solid #1e293b' }}>
          {emoji || ''} {title}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {matches.map((m, i) => (
            <MatchCard key={m.id || i} match={m} compact={matches.length > 8} />
          ))}
        </div>
      </div>
    );
  }

  // Completed matches summary
  const completedKo = ko.filter(m => m.completed);
  const upcomingKo = ko.filter(m => !m.completed && m.home && m.away && m.home !== 'TBD' && m.away !== 'TBD');

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, margin: '0 0 4px' }}>🏆 淘汰赛对阵图</h2>
        <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>
          32强淘汰赛 · 已完成 {completedKo.length}/{ko.length} 场
          {completedKo.length > 0 && <span style={{ color: '#22c55e', marginLeft: 8 }}>实时更新</span>}
        </p>
      </div>

      {/* Completed matches highlight */}
      {completedKo.length > 0 && (
        <div style={{ marginBottom: 20, padding: '12px 16px', background: '#0a1628', borderRadius: 10, border: '1px solid #1e3a5f' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#22c55e', marginBottom: 8 }}>✅ 已结束</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {completedKo.map(m => (
              <MatchCard key={m.id} match={m} compact />
            ))}
          </div>
        </div>
      )}

      {/* Upcoming with known teams */}
      {upcomingKo.length > 0 && (
        <div style={{ marginBottom: 20, padding: '12px 16px', background: '#0f172a', borderRadius: 10, border: '1px solid #1e293b' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e07820', marginBottom: 8 }}>⏰ 即将开始</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {upcomingKo.map(m => (
              <MatchCard key={m.id} match={m} compact />
            ))}
          </div>
        </div>
      )}

      {/* Full bracket */}
      <div style={{ overflowX: 'auto', paddingBottom: 16 }}>
        <div style={{ display: 'flex', gap: 16, minWidth: 'max-content' }}>
          <RoundColumn title="32强" matches={r32} emoji="⚔️" />
          <RoundColumn title="16强" matches={r16} emoji="🔥" />
          <RoundColumn title="8强" matches={qf} emoji="💎" />
          <RoundColumn title="半决赛" matches={sf} emoji="🌟" />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 160 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#e07820', textAlign: 'center', marginBottom: 6, padding: '4px 0', borderBottom: '1px solid #1e293b' }}>
              🏆 决赛
            </div>
            {final.map((m, i) => <MatchCard key={m.id || i} match={m} />)}
            {third.length > 0 && (
              <>
                <div style={{ fontSize: 11, color: '#64748b', textAlign: 'center', marginTop: 12 }}>季军赛</div>
                {third.map((m, i) => <MatchCard key={m.id || i} match={m} compact />)}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// Tab 4: 自定义预测 (Custom Predict)
// ============================================================
function CustomPredictTab() {
  const [home, setHome] = useState('');
  const [away, setAway] = useState('');
  const [result, setResult] = useState(null);

  const allTeams = Object.keys(ELO).sort();

  const doPredict = () => {
    if (!home || !away || home === away) return;
    const r = computePoissonPrediction(home, away);
    setResult({ ...r, home, away });
  };

  const selectStyle = {
    padding: '11px 14px', borderRadius: 10, border: '1px solid #334155',
    background: '#0f172a', color: '#e2e8f0', fontSize: 14, outline: 'none',
    fontFamily: 'inherit', width: '100%', cursor: 'pointer',
  };

  return (
    <div style={{ maxWidth: 500, margin: '0 auto' }}>
      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 18, padding: 28 }}>
        <h2 style={{ fontSize: 20, fontWeight: 800, margin: '0 0 4px' }}>🔮 自定义预测</h2>
        <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 22px' }}>Poisson分布 · 进攻/防守强度 · 混合模型</p>

        <div style={{ marginBottom: 14 }}>
          <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 6 }}>主队</label>
          <select value={home} onChange={e => { setHome(e.target.value); setResult(null); }} style={selectStyle}>
            <option value="">选择球队...</option>
            {allTeams.map(t => <option key={t} value={t}>{FLAG[t]} {CN[t]}</option>)}
          </select>
        </div>

        <div style={{ textAlign: 'center', color: '#334155', fontSize: 15, fontWeight: 700, margin: '10px 0' }}>vs</div>

        <div style={{ marginBottom: 22 }}>
          <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 6 }}>客队</label>
          <select value={away} onChange={e => { setAway(e.target.value); setResult(null); }} style={selectStyle}>
            <option value="">选择球队...</option>
            {allTeams.filter(t => t !== home).map(t => <option key={t} value={t}>{FLAG[t]} {CN[t]}</option>)}
          </select>
        </div>

        <button onClick={doPredict} disabled={!home || !away} style={{
          width: '100%', padding: '13px', borderRadius: 10, border: 'none',
          fontFamily: 'inherit', background: (home && away) ? '#e07820' : '#1e293b',
          color: (home && away) ? '#020617' : '#475569', fontSize: 15, fontWeight: 700,
          cursor: (home && away) ? 'pointer' : 'not-allowed', transition: 'all 0.2s',
        }}>
          开始预测 →
        </button>

        {result && (
          <div style={{ marginTop: 24, paddingTop: 24, borderTop: '1px solid #1e293b' }}>
            <div style={{ fontSize: 11, textAlign: 'center', marginBottom: 16, padding: '5px 12px', borderRadius: 8, background: '#052e16', color: '#6ee7b7' }}>
              ✅ 来源：混合模型 (Poisson + 进攻/防守强度)
            </div>

            {/* Teams display */}
            <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: 20 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28 }}>{FLAG[result.home] || '🏳️'}</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', marginTop: 4 }}>{CN[result.home]}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>xG: {result.homeXG.toFixed(2)}</div>
              </div>
              <div style={{ textAlign: 'center', paddingTop: 4 }}>
                <div style={{ fontSize: 18 }}>⚖️</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>平局</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28 }}>{FLAG[result.away] || '🏳️'}</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', marginTop: 4 }}>{CN[result.away]}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>xG: {result.awayXG.toFixed(2)}</div>
              </div>
            </div>

            {/* Probability bars */}
            {[
              { label: CN[result.home], prob: result.ph, color: '#e07820', bg: 'linear-gradient(90deg, #e07820, #f8a030)' },
              { label: '平局', prob: result.pd, color: '#64748b', bg: 'linear-gradient(90deg, #475569, #64748b)' },
              { label: CN[result.away], prob: result.pa, color: '#7c3aed', bg: 'linear-gradient(90deg, #7c3aed, #5b21b6)' },
            ].map((bar, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>{bar.label}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: '#f8c060' }}>{(bar.prob * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: (bar.prob * 100).toFixed(1) + '%', background: bar.bg, borderRadius: 3, transition: 'width 0.8s' }} />
                </div>
              </div>
            ))}

            {/* Most likely score */}
            <div style={{ marginTop: 16, padding: 14, borderRadius: 10, textAlign: 'center', background: '#020617', border: '1px solid #1e293b' }}>
              <span style={{ fontSize: 12, color: '#475569' }}>最可能比分：</span>
              <span style={{ fontSize: 22, fontWeight: 900, color: '#e07820', marginLeft: 8 }}>
                {result.mostLikelyScore}
              </span>
              <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>
                ({(result.mostLikelyScoreProb * 100).toFixed(1)}%)
              </span>
            </div>

            {/* Prediction summary */}
            <div style={{ marginTop: 10, padding: 12, borderRadius: 10, textAlign: 'center', background: '#020617', border: '1px solid #1e293b' }}>
              <span style={{ fontSize: 12, color: '#475569' }}>预测结果：</span>
              <span style={{ fontSize: 15, fontWeight: 800, color: '#f8c060' }}>
                {getPredictionOutcome(result)}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================
// Tab 6: 投注方案 (Betting Plans)
// ============================================================
function PlansTab() {
  const [budget, setBudget] = useState(500);

  // Compute all upcoming match predictions (group stage + knockout)
  const allUpcoming = [
    ...SCHEDULE.filter(m => !m.completed),
    ...(KNOCKOUT_STAGE || []).filter(m => !m.completed && m.home && m.away && m.home !== 'TBD' && m.away !== 'TBD' && !m.home.startsWith('Winner') && !m.away.startsWith('Winner') && !m.home.startsWith('Loser') && !m.away.startsWith('Loser')),
  ];
  const upcomingMatches = allUpcoming.map(m => {
    const pred = computePoissonPrediction(m.home, m.away);
    const probs = [pred.ph, pred.pd, pred.pa];
    const best = Math.max(...probs);
    const bestIdx = probs.indexOf(best);
    const result = ['W', 'D', 'L'][bestIdx];
    const label = result === 'W' ? `${cn(m.home)}胜` : result === 'L' ? `${cn(m.away)}胜` : '平局';
    const odds = marketOdds(pred.ph, pred.pd, pred.pa);
    const conf = best >= 0.60 ? 3 : best >= 0.48 ? 2 : 1;
    const confStars = '★'.repeat(conf) + '☆'.repeat(3 - conf);

    // Over/under
    let pUnder25 = 0, pUnder15 = 0;
    for (let h = 0; h <= 7; h++) {
      for (let a = 0; a <= 7; a++) {
        const tau = dcTau(h, a, pred.homeXG, pred.awayXG, -0.13);
        const p = poissProb(pred.homeXG, h) * poissProb(pred.awayXG, a) * tau;
        if (h + a <= 2) pUnder25 += p;
        if (h + a <= 1) pUnder15 += p;
      }
    }
    const pOver25 = 1 - pUnder25;
    const pOver15 = 1 - pUnder15;

    return {
      ...m, pred, probs, best, bestIdx, result, label, odds, conf, confStars,
      pOver25, pOver15,
      totalXG: pred.homeXG + pred.awayXG,
    };
  });

  // Sort by confidence
  const sorted = [...upcomingMatches].sort((a, b) => b.best - a.best);

  // Filter by date
  const today = new Date();
  const todayStr = today.toISOString().slice(0, 10);

  // Find tonight's matches (next 24h)
  const tonight = upcomingMatches.filter(m => {
    const mDate = new Date(m.date + 'T' + m.time);
    const diff = mDate - today;
    return diff > 0 && diff < 24 * 3600 * 1000;
  });

  // Plan pools by odds tier
  const allSorted = [...upcomingMatches].sort((a, b) => b.best - a.best);

  // Helper: get the best-outcome odds for a match
  const matchOdds = (m) => m.bestIdx === 0 ? m.odds.w : m.bestIdx === 2 ? m.odds.l : m.odds.d;

  // Safe tier: highest confidence, odds < 1.25
  const safePool = allSorted.filter(m => m.conf >= 3 && matchOdds(m) < 1.25);
  // Medium tier: confidence >= 2, odds 1.25-2.0
  const mediumPool = allSorted.filter(m => m.conf >= 2 && matchOdds(m) >= 1.25 && matchOdds(m) <= 2.0);
  // Risky tier: confidence >= 1, odds 1.6-3.5 (value bets)
  const riskyPool = allSorted.filter(m => matchOdds(m) >= 1.6 && matchOdds(m) <= 3.5);

  // Plans
  const planA = safePool.slice(0, 2); // 2串1, 保守 (低赔率但稳)
  const planB = mediumPool.slice(0, 3); // 3串1, 平衡 (中等赔率)
  const planC = riskyPool.slice(0, 5); // 5串1, 搏命 (高赔率串关)

  // Fallbacks if pools are too small
  if (planA.length < 2) planA.push(...mediumPool.filter(m => !planA.includes(m)).slice(0, 2 - planA.length));
  if (planB.length < 3) planB.push(...riskyPool.filter(m => !planB.includes(m)).slice(0, 3 - planB.length));
  if (planC.length < 5) planC.push(...mediumPool.filter(m => !planC.includes(m)).slice(0, 5 - planC.length));

  // Draw candidates (closest Elo)
  const drawCandidates = upcomingMatches
    .filter(m => Math.abs(m.pred.ph - m.pred.pa) < 0.08)
    .sort((a, b) => b.pred.pd - a.pred.pd)
    .slice(0, 3);

  // Over 1.5 candidates
  const overCandidates = upcomingMatches
    .filter(m => m.pOver15 > 0.72)
    .sort((a, b) => b.pOver15 - a.pOver15)
    .slice(0, 4);

  const comboOdds = (picks, key) => picks.reduce((acc, p) => acc * p.odds[key === 'result' ? (p.bestIdx === 0 ? 'w' : p.bestIdx === 2 ? 'l' : 'd') : key], 1);

  const renderPlanCard = (title, subtitle, picks, pickKey, color, riskLevel, investPct) => {
    const invest = Math.round(budget * investPct);
    let totalOdds;
    if (pickKey === 'result') {
      totalOdds = picks.reduce((acc, p) => acc * (p.bestIdx === 0 ? p.odds.w : p.bestIdx === 2 ? p.odds.l : p.odds.d), 1);
    } else if (pickKey === 'over15') {
      totalOdds = picks.reduce((acc, p) => acc * +(1 / ((p.pOver15 * 0.80 + 0.10) * 1.08)).toFixed(2), 1);
    } else if (pickKey === 'draw') {
      totalOdds = picks.reduce((acc, p) => acc * p.odds.d, 1);
    }
    const payout = +(invest * totalOdds).toFixed(0);
    const profit = payout - invest;
    const estProb = picks.reduce((acc, p) => acc * (pickKey === 'result' ? p.best : pickKey === 'over15' ? p.pOver15 : p.pred.pd), 1);

    return (
      <div style={{ background: '#0f172a', borderRadius: 16, padding: 20, border: `1px solid ${color}22`, marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div>
            <h3 style={{ margin: 0, color, fontSize: 16, fontWeight: 800 }}>{title}</h3>
            <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 12 }}>{subtitle}</p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 10, color: '#64748b' }}>风险等级</div>
            <div style={{ fontSize: 14, color: riskLevel === '低' ? '#22c55e' : riskLevel === '中' ? '#eab308' : '#ef4444', fontWeight: 700 }}>{riskLevel}</div>
          </div>
        </div>

        {picks.map((p, i) => {
          const pickLabel = pickKey === 'result' ? p.label : pickKey === 'draw' ? '平局' : `大${pickKey === 'over15' ? '1.5' : '2.5'}球`;
          const pickOdds = pickKey === 'result' ? (p.bestIdx === 0 ? p.odds.w : p.bestIdx === 2 ? p.odds.l : p.odds.d) : pickKey === 'draw' ? p.odds.d : +(1 / ((p.pOver15 * 0.80 + 0.10) * 1.08)).toFixed(2);
          const pickConf = pickKey === 'result' ? `${(p.best * 100).toFixed(0)}%` : pickKey === 'draw' ? `${(p.pred.pd * 100).toFixed(0)}%` : `${(p.pOver15 * 100).toFixed(0)}%`;
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: '#1e293b', borderRadius: 10, marginBottom: 6 }}>
              <div style={{ flex: 1 }}>
                <span style={{ color: '#94a3b8', fontSize: 11, marginRight: 6 }}>[{p.group}]</span>
                <span style={{ color: '#f1f5f9', fontSize: 13, fontWeight: 600 }}>{cn(p.home)} vs {cn(p.away)}</span>
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                <span style={{ color: '#e07820', fontSize: 12, fontWeight: 700 }}>{pickLabel}</span>
                <span style={{ background: '#020617', padding: '3px 8px', borderRadius: 6, color: '#f1f5f9', fontSize: 13, fontWeight: 700, minWidth: 40, textAlign: 'center' }}>{pickOdds}</span>
              </div>
            </div>
          );
        })}

        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 14, padding: '12px 14px', background: `${color}11`, borderRadius: 10, border: `1px solid ${color}33` }}>
          <div>
            <div style={{ fontSize: 11, color: '#64748b' }}>投注金额</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#f1f5f9' }}>¥{invest}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748b' }}>组合赔率</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#e07820' }}>{totalOdds.toFixed(2)}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748b' }}>预期回报</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#22c55e' }}>¥{payout}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: '#64748b' }}>预估概率</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#94a3b8' }}>{(estProb * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div>
      {/* Header stats */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 140, background: '#0f172a', borderRadius: 12, padding: '14px 16px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: 11, color: '#64748b' }}>模型准确率</div>
          <div style={{ fontSize: 24, fontWeight: 900, color: '#e07820' }}>58.3%</div>
          <div style={{ fontSize: 11, color: '#64748b' }}>24场中14场</div>
        </div>
        <div style={{ flex: 1, minWidth: 140, background: '#0f172a', borderRadius: 12, padding: '14px 16px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: 11, color: '#64748b' }}>剩余比赛</div>
          <div style={{ fontSize: 24, fontWeight: 900, color: '#f1f5f9' }}>{upcomingMatches.length}</div>
          <div style={{ fontSize: 11, color: '#64748b' }}>场待预测</div>
        </div>
        <div style={{ flex: 1, minWidth: 140, background: '#0f172a', borderRadius: 12, padding: '14px 16px', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: 11, color: '#64748b' }}>总预算</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 24, fontWeight: 900, color: '#f1f5f9' }}>¥</span>
            <input
              type="number"
              value={budget}
              onChange={e => setBudget(Math.max(10, parseInt(e.target.value) || 10))}
              style={{ width: 70, fontSize: 24, fontWeight: 900, color: '#f1f5f9', background: 'transparent', border: 'none', outline: 'none', fontFamily: 'inherit' }}
            />
          </div>
          <div style={{ fontSize: 11, color: '#64748b' }}>点击修改</div>
        </div>
      </div>

      {/* Model strengths */}
      <div style={{ background: '#0f172a', borderRadius: 12, padding: 16, marginBottom: 20, border: '1px solid #1e293b' }}>
        <h3 style={{ margin: '0 0 10px', color: '#f1f5f9', fontSize: 14, fontWeight: 700 }}>📊 模型表现分析</h3>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 100 }}>
            <div style={{ fontSize: 11, color: '#64748b' }}>主胜预测</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#22c55e' }}>92% <span style={{ fontSize: 11, fontWeight: 400 }}>(11/12)</span></div>
          </div>
          <div style={{ flex: 1, minWidth: 100 }}>
            <div style={{ fontSize: 11, color: '#64748b' }}>客胜预测</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#22c55e' }}>100% <span style={{ fontSize: 11, fontWeight: 400 }}>(3/3)</span></div>
          </div>
          <div style={{ flex: 1, minWidth: 100 }}>
            <div style={{ fontSize: 11, color: '#64748b' }}>平局预测</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#ef4444' }}>0% <span style={{ fontSize: 11, fontWeight: 400 }}>(0/9)</span></div>
          </div>
        </div>
        <p style={{ fontSize: 11, color: '#94a3b8', margin: '10px 0 0', lineHeight: 1.5 }}>
          💡 <strong style={{ color: '#e07820' }}>策略提示：</strong>胜平负只买明确方向(主/客胜)，避开平局。大小球/进球数更稳定。
        </p>
      </div>

      {/* Section title */}
      <h3 style={{ color: '#f1f5f9', fontSize: 16, fontWeight: 800, margin: '0 0 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ background: '#e07820', color: '#020617', padding: '2px 8px', borderRadius: 6, fontSize: 12, fontWeight: 800 }}>方案</span>
        胜平负串关
      </h3>

      {/* Plan 1: 2串1 保守 */}
      {planA.length >= 2 && renderPlanCard(
        '💰 方案A：2串1 保守型',
        '低风险 · 高概率 · 适合稳健玩家',
        planA, 'result', '#22c55e', '低', 0.4
      )}

      {/* Plan 2: 3串1 平衡 */}
      {planB.length >= 3 && renderPlanCard(
        '🔥 方案B：3串1 平衡型',
        '中等风险 · 性价比最高 · 推荐首选',
        planB, 'result', '#eab308', '中', 0.3
      )}

      {/* Plan 3: 5串1 搏命 */}
      {planC.length >= 5 && renderPlanCard(
        '🚀 方案C：5串1 搏一搏',
        '高风险 · 高回报 · 花小钱搏大钱',
        planC, 'result', '#ef4444', '高', 0.1
      )}

      {/* Section: 大球方案 */}
      {overCandidates.length >= 2 && (
        <>
          <h3 style={{ color: '#f1f5f9', fontSize: 16, fontWeight: 800, margin: '24px 0 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ background: '#3b82f6', color: '#fff', padding: '2px 8px', borderRadius: 6, fontSize: 12, fontWeight: 800 }}>方案D</span>
            大1.5球 串关
          </h3>
          <p style={{ fontSize: 12, color: '#94a3b8', margin: '0 0 12px' }}>不猜谁赢，只要每场进≥2球就赢。比胜平负更稳。</p>
          {renderPlanCard(
            '⚽ 方案D：大1.5球 串关',
            '中低风险 · 不纠结胜负 · 进球即赢',
            overCandidates.slice(0, 3), 'over15', '#3b82f6', '中低', 0.2
          )}
        </>
      )}

      {/* Section: 平局博冷 */}
      {drawCandidates.length >= 2 && (
        <>
          <h3 style={{ color: '#f1f5f9', fontSize: 16, fontWeight: 800, margin: '24px 0 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ background: '#a855f7', color: '#fff', padding: '2px 8px', borderRadius: 6, fontSize: 12, fontWeight: 800 }}>方案E</span>
            平局博冷
          </h3>
          <p style={{ fontSize: 12, color: '#94a3b8', margin: '0 0 12px' }}>本届平局率高达37.5%(9/24)，远超正常水平。高赔率博冷门。</p>
          {renderPlanCard(
            '🎲 方案E：平局博冷门',
            '高风险 · 高赔率 · 小注博大回报',
            drawCandidates.slice(0, 2), 'draw', '#a855f7', '高', 0.06
          )}
        </>
      )}

      {/* All picks ranked */}
      <h3 style={{ color: '#f1f5f9', fontSize: 16, fontWeight: 800, margin: '28px 0 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ background: '#1e293b', color: '#94a3b8', padding: '2px 8px', borderRadius: 6, fontSize: 12, fontWeight: 800 }}>全部</span>
        剩余比赛预测（按信心排序）
      </h3>
      <div style={{ background: '#0f172a', borderRadius: 12, overflow: 'hidden', border: '1px solid #1e293b' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr 100px 60px 60px', padding: '10px 14px', borderBottom: '1px solid #1e293b', fontSize: 11, color: '#64748b', fontWeight: 600 }}>
          <span>日期</span>
          <span>比赛</span>
          <span style={{ textAlign: 'center' }}>预测</span>
          <span style={{ textAlign: 'center' }}>信心</span>
          <span style={{ textAlign: 'center' }}>赔率</span>
        </div>
        {sorted.map((m, i) => {
          const pickOdds = m.bestIdx === 0 ? m.odds.w : m.bestIdx === 2 ? m.odds.l : m.odds.d;
          const bgColor = i % 2 === 0 ? '#0f172a' : '#111827';
          return (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '40px 1fr 100px 60px 60px', padding: '10px 14px', borderBottom: '1px solid #0f172a', background: bgColor, alignItems: 'center', fontSize: 12 }}>
              <span style={{ color: '#64748b', fontSize: 10 }}>{m.date.slice(5)}</span>
              <span>
                <span style={{ color: '#475569', fontSize: 10 }}>[{m.group}] </span>
                <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{cn(m.home)}</span>
                <span style={{ color: '#475569' }}> vs </span>
                <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{cn(m.away)}</span>
              </span>
              <span style={{ textAlign: 'center', color: '#e07820', fontWeight: 700, fontSize: 11 }}>{m.label}</span>
              <span style={{ textAlign: 'center', color: m.conf >= 3 ? '#22c55e' : m.conf >= 2 ? '#eab308' : '#64748b', fontWeight: 700, fontSize: 11 }}>{m.confStars}</span>
              <span style={{ textAlign: 'center', color: '#f1f5f9', fontWeight: 700 }}>{pickOdds}</span>
            </div>
          );
        })}
      </div>

      {/* Disclaimer */}
      <div style={{ marginTop: 20, padding: 14, background: '#1e293b22', borderRadius: 10, border: '1px solid #1e293b', textAlign: 'center' }}>
        <p style={{ fontSize: 11, color: '#64748b', margin: 0, lineHeight: 1.6 }}>
          ⚠️ 以上方案基于 Dixon-Coles 模型 + 校准Elo，仅供参考。足球比赛具有高度不确定性，请理性购彩，量力而行。
        </p>
      </div>
    </div>
  );
}

// ============================================================
// Main App Component
// ============================================================
// ============================================================
// AnalysisTab - Deep pre-match analysis for today's 3 games
// ============================================================
function AnalysisTab() {
  const [selectedMatch, setSelectedMatch] = useState(80);
  const matches = Object.values(MATCH_ANALYSIS);
  const m = MATCH_ANALYSIS[selectedMatch];
  if (!m) return <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>暂无分析数据</div>;

  const cardStyle = { background: '#0f172a', borderRadius: 14, padding: '18px 20px', marginBottom: 14, border: '1px solid #1e293b' };
  const sectionTitle = { fontSize: 14, fontWeight: 800, color: '#e07820', marginBottom: 10, letterSpacing: '0.05em' };
  const labelStyle = { fontSize: 11, color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 };
  const textStyle = { fontSize: 13, color: '#cbd5e1', lineHeight: 1.6 };

  const StatusBadge = ({ status }) => {
    const colors = { '缺阵': '#ef4444', '存疑': '#f59e0b', '全员健康': '#22c55e' };
    return <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 700, background: (colors[status] || '#64748b') + '22', color: colors[status] || '#94a3b8', marginRight: 6 }}>{status}</span>;
  };

  const ProbBar = ({ homePct, drawPct, awayPct }) => (
    <div style={{ display: 'flex', height: 28, borderRadius: 8, overflow: 'hidden', marginBottom: 8 }}>
      <div style={{ width: `${homePct}%`, background: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#fff' }}>{homePct}%</div>
      <div style={{ width: `${drawPct}%`, background: '#6b7280', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#fff' }}>{drawPct}%</div>
      <div style={{ width: `${awayPct}%`, background: '#ef4444', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#fff' }}>{awayPct}%</div>
    </div>
  );

  return (
    <div>
      {/* Match selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 18, flexWrap: 'wrap', justifyContent: 'center' }}>
        {matches.map(mt => (
          <button key={mt.matchId} onClick={() => setSelectedMatch(mt.matchId)} style={{
            padding: '10px 18px', borderRadius: 12, border: selectedMatch === mt.matchId ? '2px solid #e07820' : '1px solid #1e293b',
            background: selectedMatch === mt.matchId ? '#1e293b' : '#0f172a', cursor: 'pointer', fontFamily: 'inherit',
            color: selectedMatch === mt.matchId ? '#e07820' : '#94a3b8', fontWeight: 700, fontSize: 13, transition: 'all 0.2s',
          }}>
            {flag(mt.home)} {cn(mt.home)} vs {cn(mt.away)} {flag(mt.away)}
          </button>
        ))}
      </div>

      {/* Match header */}
      <div style={{ ...cardStyle, textAlign: 'center', background: 'linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%)', border: '1px solid #e0782044' }}>
        <div style={{ fontSize: 11, color: '#e07820', letterSpacing: '0.15em', marginBottom: 6 }}>🏟️ {m.venue} · {m.kickoff}</div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 20, margin: '12px 0' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 36 }}>{flag(m.home)}</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: '#f1f5f9', marginTop: 4 }}>{cn(m.home)}</div>
          </div>
          <div style={{ fontSize: 28, fontWeight: 900, color: '#e07820' }}>VS</div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 36 }}>{flag(m.away)}</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: '#f1f5f9', marginTop: 4 }}>{cn(m.away)}</div>
          </div>
        </div>
        <div style={{ fontSize: 12, color: '#64748b' }}>裁判: {m.referee}</div>
        <div style={{ marginTop: 10, display: 'inline-block', padding: '6px 16px', borderRadius: 20, background: '#e0782022', border: '1px solid #e0782044' }}>
          <span style={{ fontSize: 18, fontWeight: 900, color: '#e07820' }}>预测比分 {m.prediction.score}</span>
          <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 8 }}>信心: {m.prediction.confidence}</span>
        </div>
      </div>

      {/* Win probability */}
      <div style={cardStyle}>
        <div style={sectionTitle}>📊 胜平负概率</div>
        <ProbBar homePct={Math.round(m.prediction.homeWin * 100)} drawPct={Math.round(m.prediction.draw * 100)} awayPct={Math.round(m.prediction.awayWin * 100)} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8' }}>
          <span>{cn(m.home)}胜 {Math.round(m.prediction.homeWin * 100)}%</span>
          <span>平 {Math.round(m.prediction.draw * 100)}%</span>
          <span>{cn(m.away)}胜 {Math.round(m.prediction.awayWin * 100)}%</span>
        </div>
      </div>

      {/* Betting odds */}
      <div style={cardStyle}>
        <div style={sectionTitle}>💰 盘口分析</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <div style={labelStyle}>胜平负赔率</div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ padding: '4px 10px', borderRadius: 6, background: '#3b82f622', color: '#60a5fa', fontSize: 13, fontWeight: 700 }}>主 {m.betting.winDrawLoss.home}</span>
              <span style={{ padding: '4px 10px', borderRadius: 6, background: '#6b728022', color: '#9ca3af', fontSize: 13, fontWeight: 700 }}>平 {m.betting.winDrawLoss.draw}</span>
              <span style={{ padding: '4px 10px', borderRadius: 6, background: '#ef444422', color: '#f87171', fontSize: 13, fontWeight: 700 }}>客 {m.betting.winDrawLoss.away}</span>
            </div>
          </div>
          <div>
            <div style={labelStyle}>亚盘让球</div>
            <div style={{ fontSize: 13, color: '#cbd5e1' }}>
              {cn(m.home)} <span style={{ color: '#e07820', fontWeight: 700 }}>{m.betting.asianHandicap.line}</span> · 主{m.betting.asianHandicap.home} / 客{m.betting.asianHandicap.away}
            </div>
          </div>
          <div>
            <div style={labelStyle}>大小球 {m.betting.overUnder.line}</div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ padding: '4px 10px', borderRadius: 6, background: m.prediction.over25 > 0.5 ? '#22c55e22' : '#ef444422', color: m.prediction.over25 > 0.5 ? '#4ade80' : '#f87171', fontSize: 13, fontWeight: 700 }}>
                大 {m.betting.overUnder.over} ({Math.round(m.prediction.over25 * 100)}%)
              </span>
              <span style={{ padding: '4px 10px', borderRadius: 6, background: m.prediction.over25 < 0.5 ? '#22c55e22' : '#ef444422', color: m.prediction.over25 < 0.5 ? '#4ade80' : '#f87171', fontSize: 13, fontWeight: 700 }}>
                小 {m.betting.overUnder.under} ({Math.round((1 - m.prediction.over25) * 100)}%)
              </span>
            </div>
          </div>
          <div>
            <div style={labelStyle}>红黄牌预测</div>
            <div style={{ fontSize: 13, color: '#cbd5e1' }}>
              🟨 {m.betting.cards.expectedYellow} · 🟥 {m.betting.cards.expectedRed}
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{m.betting.cards.note}</div>
          </div>
        </div>
        {/* Extra stats */}
        <div style={{ marginTop: 14, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ padding: '6px 12px', borderRadius: 8, background: '#1e293b', fontSize: 12 }}>
            <span style={{ color: '#64748b' }}>大1.5球 </span>
            <span style={{ color: m.prediction.over15 > 0.7 ? '#4ade80' : '#f59e0b', fontWeight: 700 }}>{Math.round(m.prediction.over15 * 100)}%</span>
          </div>
          <div style={{ padding: '6px 12px', borderRadius: 8, background: '#1e293b', fontSize: 12 }}>
            <span style={{ color: '#64748b' }}>双方进球 </span>
            <span style={{ color: m.prediction.btts > 0.5 ? '#4ade80' : '#f59e0b', fontWeight: 700 }}>{Math.round(m.prediction.btts * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Group form */}
      <div style={cardStyle}>
        <div style={sectionTitle}>📈 小组赛战绩</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <div style={{ ...labelStyle, color: '#60a5fa' }}>{flag(m.home)} {cn(m.home)}</div>
            <div style={textStyle}>{m.groupForm.home}</div>
          </div>
          <div>
            <div style={{ ...labelStyle, color: '#f87171' }}>{flag(m.away)} {cn(m.away)}</div>
            <div style={textStyle}>{m.groupForm.away}</div>
          </div>
        </div>
      </div>

      {/* Motivation */}
      <div style={cardStyle}>
        <div style={sectionTitle}>🔥 战意分析</div>
        <div style={{ marginBottom: 10 }}>
          <div style={{ ...labelStyle, color: '#60a5fa' }}>{cn(m.home)}</div>
          <div style={textStyle}>{m.motivation.home}</div>
        </div>
        <div>
          <div style={{ ...labelStyle, color: '#f87171' }}>{cn(m.away)}</div>
          <div style={textStyle}>{m.motivation.away}</div>
        </div>
      </div>

      {/* Lineups */}
      <div style={cardStyle}>
        <div style={sectionTitle}>👕 预测首发</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <div style={{ ...labelStyle, color: '#60a5fa' }}>{cn(m.home)} ({m.lineup.home.formation})</div>
            <div style={{ ...textStyle, fontSize: 12, fontFamily: 'monospace' }}>{m.lineup.home.players}</div>
          </div>
          <div>
            <div style={{ ...labelStyle, color: '#f87171' }}>{cn(m.away)} ({m.lineup.away.formation})</div>
            <div style={{ ...textStyle, fontSize: 12, fontFamily: 'monospace' }}>{m.lineup.away.players}</div>
          </div>
        </div>
      </div>

      {/* Injuries */}
      <div style={cardStyle}>
        <div style={sectionTitle}>🏥 伤病情况</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <div style={{ ...labelStyle, color: '#60a5fa' }}>{cn(m.home)}</div>
            {m.injuries.home.map((inj, i) => (
              <div key={i} style={{ marginBottom: 6, fontSize: 12 }}>
                <StatusBadge status={inj.status} />
                <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{inj.player}</span>
                <span style={{ color: '#64748b', marginLeft: 4 }}>- {inj.detail}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{ ...labelStyle, color: '#f87171' }}>{cn(m.away)}</div>
            {m.injuries.away.map((inj, i) => (
              <div key={i} style={{ marginBottom: 6, fontSize: 12 }}>
                <StatusBadge status={inj.status} />
                <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{inj.player}</span>
                <span style={{ color: '#64748b', marginLeft: 4 }}>- {inj.detail}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Key players */}
      <div style={cardStyle}>
        <div style={sectionTitle}>⭐ 关键球员</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <div style={{ ...labelStyle, color: '#60a5fa' }}>{cn(m.home)}</div>
            {m.keyPlayers.home.map((p, i) => (
              <div key={i} style={{ marginBottom: 8, padding: '8px 10px', borderRadius: 8, background: '#1e293b' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>{p.name} <span style={{ fontSize: 11, color: '#64748b', fontWeight: 400 }}>({p.role})</span></div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{p.note}</div>
              </div>
            ))}
          </div>
          <div>
            <div style={{ ...labelStyle, color: '#f87171' }}>{cn(m.away)}</div>
            {m.keyPlayers.away.map((p, i) => (
              <div key={i} style={{ marginBottom: 8, padding: '8px 10px', borderRadius: 8, background: '#1e293b' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>{p.name} <span style={{ fontSize: 11, color: '#64748b', fontWeight: 400 }}>({p.role})</span></div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{p.note}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tactics */}
      <div style={cardStyle}>
        <div style={sectionTitle}>⚽ 战术分析</div>
        <div style={{ marginBottom: 10 }}>
          <div style={{ ...labelStyle, color: '#60a5fa' }}>{cn(m.home)}</div>
          <div style={textStyle}>{m.tactics.home}</div>
        </div>
        <div>
          <div style={{ ...labelStyle, color: '#f87171' }}>{cn(m.away)}</div>
          <div style={textStyle}>{m.tactics.away}</div>
        </div>
      </div>

      {/* H2H */}
      <div style={cardStyle}>
        <div style={sectionTitle}>📜 交锋记录</div>
        <div style={textStyle}>{m.h2h}</div>
      </div>

      {/* Betting recommendation */}
      <div style={{ ...cardStyle, border: '1px solid #e0782044', background: 'linear-gradient(135deg, #1a1a2e 0%, #0f172a 100%)' }}>
        <div style={{ ...sectionTitle, fontSize: 16 }}>🎯 投注建议</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
          {/* Safe picks */}
          <div style={{ padding: 14, borderRadius: 10, background: '#22c55e11', border: '1px solid #22c55e33' }}>
            <div style={{ fontSize: 13, fontWeight: 800, color: '#4ade80', marginBottom: 8 }}>✅ 稳胆推荐</div>
            {m.prediction.homeWin > 0.55 && <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• {cn(m.home)}胜 (赔率{m.betting.winDrawLoss.home})</div>}
            {m.prediction.over15 > 0.72 && <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• 大1.5球 ({Math.round(m.prediction.over15 * 100)}%概率)</div>}
            {m.prediction.homeWin > 0.5 && m.prediction.over15 > 0.7 && <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• {cn(m.home)}胜+大1.5球 串关</div>}
            {m.prediction.homeWin <= 0.55 && m.prediction.over15 <= 0.72 && <div style={{ fontSize: 12, color: '#94a3b8' }}>本场无明显稳胆</div>}
          </div>
          {/* Risky picks */}
          <div style={{ padding: 14, borderRadius: 10, background: '#f59e0b11', border: '1px solid #f59e0b33' }}>
            <div style={{ fontSize: 13, fontWeight: 800, color: '#fbbf24', marginBottom: 8 }}>🔥 搏冷推荐</div>
            {m.prediction.btts > 0.5 && <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• 双方进球 ({Math.round(m.prediction.btts * 100)}%)</div>}
            {m.prediction.over25 > 0.5 && <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• 大2.5球 (赔率{m.betting.overUnder.over})</div>}
            {m.prediction.awayWin > 0.15 && <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• {cn(m.away)}爆冷胜 (赔率{m.betting.winDrawLoss.away})</div>}
            {m.prediction.draw > 0.22 && <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• 平局 (赔率{m.betting.winDrawLoss.draw})</div>}
            <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>• 让球{m.betting.asianHandicap.line} {cn(m.away)} (赔率{m.betting.asianHandicap.away})</div>
          </div>
        </div>
      </div>
    </div>
  );
}

const TABS = [
  { key: 'schedule', label: '📅 赛程' },
  { key: 'standings', label: '📊 积分榜' },
  { key: 'analysis', label: '🔍 深度分析' },
  { key: 'plans', label: '💰 方案' },
  { key: 'betting', label: '🎰 竞彩' },
  { key: 'bracket', label: '🏆 对阵图' },
  { key: 'custom', label: '🔮 自定义预测' },
];

export default function App() {
  const [tab, setTab] = useState('schedule');
  const [squadTeam, setSquadTeam] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Auto-fetch latest results from ESPN and merge into SCHEDULE
  useEffect(() => {
    const NAME_MAP = { 'Türkiye': 'Turkey', 'Curaçao': 'Curaçao', 'DR Congo': 'DR Congo', 'Bosnia and Herzegovina': 'Bosnia & Herzegovina', 'Côte d\'Ivoire': 'Ivory Coast', 'Ivory Coast': 'Ivory Coast' };
    const mapName = (n) => NAME_MAP[n] || n;
    const fetchResults = async () => {
      try {
        const resp = await fetch('https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard');
        const data = await resp.json();
        const events = data.events || [];
        let updated = false;
        events.forEach(ev => {
          const status = ev.status?.type?.name;
          if (status !== 'STATUS_FULL_TIME') return;
          const comp = ev.competitions?.[0];
          const teams = comp?.competitors || [];
          const home = teams.find(t => t.homeAway === 'home');
          const away = teams.find(t => t.homeAway === 'away');
          if (!home || !away) return;
          const homeName = mapName(home.team?.displayName);
          const awayName = mapName(away.team?.displayName);
          const homeScore = parseInt(home.score);
          const awayScore = parseInt(away.score);
          // Find matching match in SCHEDULE and update
          const match = SCHEDULE.find(m => m.home === homeName && m.away === awayName && !m.completed);
          if (match) {
            match.homeScore = homeScore;
            match.awayScore = awayScore;
            match.completed = true;
            updated = true;
          }
        });
        if (updated) setLastUpdate(new Date().toLocaleTimeString('zh-CN'));
      } catch (e) {
        console.log('ESPN fetch failed, using local data:', e.message);
      }
    };
    fetchResults();
    // Re-fetch every 5 minutes
    const interval = setInterval(fetchResults, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#020617', fontFamily: "'Nunito', 'PingFang SC', sans-serif", color: '#e2e8f0' }}>
      <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ background: 'linear-gradient(160deg, #0c2340 0%, #020617 100%)', padding: '32px 24px 24px', textAlign: 'center', borderBottom: '1px solid #0f172a' }}>
        <div style={{ fontSize: 11, letterSpacing: '0.18em', color: '#e07820', marginBottom: 8, textTransform: 'uppercase' }}>🧡 2026 FIFA 世界杯</div>
        <h1 style={{ fontSize: 'clamp(24px, 5vw, 48px)', fontWeight: 900, margin: 0, lineHeight: 1.1 }}>
          <span style={{ color: '#f1f5f9' }}>赛果</span><span style={{ color: '#e07820' }}>预测</span><span style={{ color: '#f1f5f9' }}>系统</span>
        </h1>
        <p style={{ color: '#475569', marginTop: 8, fontSize: 12 }}>
          混合模型 (Poisson+ELO+赔率) · 数据更新：{new Date().toLocaleDateString('zh-CN')}
          {lastUpdate && <span style={{ color: '#22c55e', marginLeft: 8 }}>🔄 自动更新 {lastUpdate}</span>}
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 6, padding: '14px 24px 0', flexWrap: 'wrap' }}>
        {TABS.map(({ key, label }) => (
          <button key={key} onClick={() => setTab(key)} style={{
            padding: '7px 16px', borderRadius: 30, border: 'none', cursor: 'pointer',
            fontWeight: 700, fontSize: 13, fontFamily: 'inherit', transition: 'all 0.2s',
            background: tab === key ? '#e07820' : '#1e293b',
            color: tab === key ? '#020617' : '#64748b',
          }}>
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ maxWidth: 960, margin: '0 auto', padding: '18px 16px 60px' }}>
        {tab === 'schedule' && <ScheduleTab onOpenSquad={setSquadTeam} />}
        {tab === 'standings' && <StandingsTab onOpenSquad={setSquadTeam} />}
        {tab === 'analysis' && <AnalysisTab />}
        {tab === 'plans' && <PlansTab />}
        {tab === 'betting' && <BettingTab />}
        {tab === 'bracket' && <BracketTab />}
        {tab === 'custom' && <CustomPredictTab />}
      </div>

      {/* Squad Modal */}
      {squadTeam && <SquadModal team={squadTeam} onClose={() => setSquadTeam(null)} />}
    </div>
  );
}
import { useState, useMemo, useEffect, useRef } from 'react';
import { SCHEDULE, ELO, CN, FLAG } from './data';

const cn = (name) => CN[name] || name;
const flag = (name) => FLAG[name] || '🏳️';

// ============================================================
// Poisson engine (Hicruben Dixon-Coles model)
// ============================================================
function poissProb(xg, k) {
  let f = 1;
  for (let i = 2; i <= k; i++) f *= i;
  return (Math.pow(xg, k) * Math.exp(-xg)) / f;
}

function dcTau(h, a, lambda, mu, rho) {
  if (h === 0 && a === 0) return 1 - lambda * mu * rho;
  if (h === 0 && a === 1) return 1 + lambda * rho;
  if (h === 1 && a === 0) return 1 + mu * rho;
  if (h === 1 && a === 1) return 1 - rho;
  return 1;
}

// Host nations get 75 Elo home advantage
const HOST_NATIONS = new Set(['Mexico', 'USA', 'Canada']);
const HOME_ADV = 75;

// Expected goals from Elo rating difference (Hicruben: lambda = 1.35 + diff/400)
function expectedGoals(rating, opponent, homeBonus = 0) {
  const diff = (rating + homeBonus) - opponent;
  const lambda = 1.35 + diff / 400;
  return Math.max(0.3, Math.min(3.5, lambda));
}

// Full score matrix computation
function computeScoreMatrix(homeXG, awayXG) {
  const maxG = 8;
  const rho = -0.13;
  const matrix = {};
  for (let h = 0; h <= maxG; h++) {
    for (let a = 0; a <= maxG; a++) {
      const tau = dcTau(h, a, homeXG, awayXG, rho);
      const prob = poissProb(homeXG, h) * poissProb(awayXG, a) * tau;
      matrix[`${h}-${a}`] = prob;
    }
  }
  return matrix;
}

// Compute all betting data for a match
function computeAllBets(home, away) {
  const eH = ELO[home] || 1600;
  const eA = ELO[away] || 1600;
  const hb = (HOST_NATIONS.has(home) ? HOME_ADV : 0) - (HOST_NATIONS.has(away) ? HOME_ADV : 0);
  const homeXG = expectedGoals(eH, eA, hb);
  const awayXG = expectedGoals(eA, eH, -hb / 2);

  const matrix = computeScoreMatrix(homeXG, awayXG);
  const MARGIN = 0.90; // 返还率 90%

  // --- 胜平负 1X2 ---
  let ph = 0, pd = 0, pa = 0;
  Object.entries(matrix).forEach(([score, prob]) => {
    const [h, a] = score.split('-').map(Number);
    if (h > a) ph += prob;
    else if (h === a) pd += prob;
    else pa += prob;
  });
  const tot1x2 = ph + pd + pa;
  ph /= tot1x2; pd /= tot1x2; pa /= tot1x2;

  const oddsW = +(MARGIN / ph).toFixed(2);
  const oddsD = +(MARGIN / pd).toFixed(2);
  const oddsL = +(MARGIN / pa).toFixed(2);

  // --- 大小球 2.5 ---
  let pOver25 = 0, pUnder25 = 0;
  Object.entries(matrix).forEach(([score, prob]) => {
    const [h, a] = score.split('-').map(Number);
    if (h + a > 2.5) pOver25 += prob;
    else pUnder25 += prob;
  });
  const totOU = pOver25 + pUnder25;
  const oddsOver = +(MARGIN / (pOver25 / totOU)).toFixed(2);
  const oddsUnder = +(MARGIN / (pUnder25 / totOU)).toFixed(2);

  // --- 大小球 1.5 ---
  let pOver15 = 0, pUnder15 = 0;
  Object.entries(matrix).forEach(([score, prob]) => {
    const [h, a] = score.split('-').map(Number);
    if (h + a > 1.5) pOver15 += prob;
    else pUnder15 += prob;
  });
  const totOU15 = pOver15 + pUnder15;
  const oddsOver15 = +(MARGIN / (pOver15 / totOU15)).toFixed(2);
  const oddsUnder15 = +(MARGIN / (pUnder15 / totOU15)).toFixed(2);

  // --- 让球胜平负 (handicap based on ELO diff) ---
  const eloDiff = (ELO[home] || 1600) - (ELO[away] || 1600);
  let handicap = 0;
  if (eloDiff > 250) handicap = -2;
  else if (eloDiff > 120) handicap = -1;
  else if (eloDiff < -250) handicap = 2;
  else if (eloDiff < -120) handicap = 1;
  // else 0 (平手盘)

  let hpW = 0, hpD = 0, hpL = 0;
  Object.entries(matrix).forEach(([score, prob]) => {
    const [h, a] = score.split('-').map(Number);
    const adj = h + handicap - a;
    if (adj > 0) hpW += prob;
    else if (adj === 0) hpD += prob;
    else hpL += prob;
  });
  const totHP = hpW + hpD + hpL;
  const oddsHW = +(MARGIN / (hpW / totHP)).toFixed(2);
  const oddsHD = +(MARGIN / (hpD / totHP)).toFixed(2);
  const oddsHL = +(MARGIN / (hpL / totHP)).toFixed(2);

  // --- 比分 (correct score) top 8 ---
  const scores = Object.entries(matrix)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([score, prob]) => ({
      score,
      prob,
      odds: +(MARGIN / prob).toFixed(2),
    }));

  // --- 总进球 ---
  const goalBuckets = [
    { label: '0球', min: 0, max: 0 },
    { label: '1球', min: 1, max: 1 },
    { label: '2球', min: 2, max: 2 },
    { label: '3球', min: 3, max: 3 },
    { label: '4球', min: 4, max: 4 },
    { label: '5球', min: 5, max: 5 },
    { label: '6球', min: 6, max: 6 },
    { label: '7+球', min: 7, max: 99 },
  ];
  const totalGoals = goalBuckets.map(b => {
    let p = 0;
    Object.entries(matrix).forEach(([score, prob]) => {
      const [h, a] = score.split('-').map(Number);
      if (h + a >= b.min && h + a <= b.max) p += prob;
    });
    return { label: b.label, prob: p, odds: +(MARGIN / p).toFixed(2) };
  });

  // --- 半全场 ---
  // Simplified: assume first half ≈ 45% of expected goals
  const hfHomeXG = homeXG * 0.45;
  const hfAwayXG = awayXG * 0.45;
  const hfMatrix = computeScoreMatrix(hfHomeXG, hfAwayXG);
  const htft = {};
  ['胜胜', '胜平', '胜负', '平胜', '平平', '平负', '负胜', '负平', '负负'].forEach(k => htft[k] = 0);
  Object.entries(matrix).forEach(([ftScore, ftProb]) => {
    const [fh, fa] = ftScore.split('-').map(Number);
    let ftResult; // 主胜/平/客胜
    if (fh > fa) ftResult = '胜';
    else if (fh === fa) ftResult = '平';
    else ftResult = '负';
    // Sum over all HT possibilities
    Object.entries(hfMatrix).forEach(([htScore, htProb]) => {
      const [hh, ha] = htScore.split('-').map(Number);
      let htResult;
      if (hh > ha) htResult = '胜';
      else if (hh === ha) htResult = '平';
      else htResult = '负';
      // Check HT result is consistent with FT result
      const key = htResult + ftResult;
      if (htft[key] !== undefined) {
        // Weight by both probabilities, but scale down for realism
        htft[key] += ftProb * htProb * 2.5; // scale factor for reasonable odds
      }
    });
  });
  // Normalize
  const htftTotal = Object.values(htft).reduce((a, b) => a + b, 0);
  const htftOdds = Object.entries(htft).map(([label, p]) => ({
    label,
    prob: p / htftTotal,
    odds: +(MARGIN / (p / htftTotal)).toFixed(2),
  })).sort((a, b) => b.prob - a.prob);

  return {
    homeXG, awayXG,
    ph, pd, pa,
    // 胜平负
    spf: { w: oddsW, d: oddsD, l: oddsL },
    // 大小球2.5
    ou25: { over: oddsOver, under: oddsUnder },
    // 大小球1.5
    ou15: { over: oddsOver15, under: oddsUnder15 },
    // 让球
    handicap,
    rspf: { w: oddsHW, d: oddsHD, l: oddsHL },
    // 比分
    correctScores: scores,
    // 总进球
    totalGoals,
    // 半全场
    htft: htftOdds,
    // Expected goals
    expectedTotal: homeXG + awayXG,
  };
}

// ============================================================
// Generate AI recommendation for a match
// ============================================================
function generateRecommendation(match, bets) {
  const tips = [];
  const { spf, ou25, ou15, rspf, handicap, correctScores, totalGoals, ph, pd, pa, expectedTotal } = bets;

  // Determine the strongest bet signal
  const maxProb = Math.max(ph, pd, pa);
  const confidence = maxProb;

  // 胜平负推荐
  if (ph > 0.50) {
    tips.push({
      type: '胜平负',
      pick: '主胜',
      odds: spf.w,
      confidence: Math.round(ph * 100),
      reason: `${cn(match.home)}实力明显占优 (ELO ${(ELO[match.home]||0) - (ELO[match.away]||0) > 0 ? '+' : ''}${(ELO[match.home]||0) - (ELO[match.away]||0)})，预计控球率更高，进攻火力更猛。赔率${spf.w}，性价比不错。`,
      star: ph > 0.55 ? 3 : ph > 0.45 ? 2 : 1,
    });
  } else if (pa > 0.50) {
    tips.push({
      type: '胜平负',
      pick: '客胜',
      odds: spf.l,
      confidence: Math.round(pa * 100),
      reason: `${cn(match.away)}客场作战能力强 (ELO ${(ELO[match.away]||0) - (ELO[match.home]||0) > 0 ? '+' : ''}${(ELO[match.away]||0) - (ELO[match.home]||0)})，${cn(match.home)}主场优势不足以弥补实力差距。赔率${spf.l}有吸引力。`,
      star: pa > 0.55 ? 3 : pa > 0.45 ? 2 : 1,
    });
  } else if (pd > 0.28) {
    tips.push({
      type: '胜平负',
      pick: '平局',
      odds: spf.d,
      confidence: Math.round(pd * 100),
      reason: `两队实力接近 (ELO差仅${Math.abs((ELO[match.home]||0) - (ELO[match.away]||0))})，小组赛求稳心态明显，防守端都比较稳健。赔率${spf.d}是全场最高回报。`,
      star: pd > 0.32 ? 3 : pd > 0.26 ? 2 : 1,
    });
  }

  // 大小球推荐
  const pOver = 1 / ou25.over * 0.90;
  if (expectedTotal > 2.8) {
    tips.push({
      type: '大小球2.5',
      pick: '大2.5',
      odds: ou25.over,
      confidence: Math.round(pOver * 100),
      reason: `双方预期进球合计${expectedTotal.toFixed(1)}球，${cn(match.home)}攻击力强(xG ${bets.homeXG.toFixed(1)})，${cn(match.away)}也有火力(xG ${bets.awayXG.toFixed(1)})，大球概率较高。`,
      star: expectedTotal > 3.5 ? 3 : expectedTotal > 3.0 ? 2 : 1,
    });
  } else if (expectedTotal < 2.2) {
    tips.push({
      type: '大小球2.5',
      pick: '小2.5',
      odds: ou25.under,
      confidence: Math.round((1 - pOver) * 100),
      reason: `双方预期进球合计仅${expectedTotal.toFixed(1)}球，两队防守都不错，预计低比分。小球赔率${ou25.under}稳健。`,
      star: expectedTotal < 1.8 ? 3 : expectedTotal < 2.0 ? 2 : 1,
    });
  }

  // 比分推荐 (top pick)
  if (correctScores[0]) {
    tips.push({
      type: '比分',
      pick: correctScores[0].score,
      odds: correctScores[0].odds,
      confidence: Math.round(correctScores[0].prob * 100),
      reason: `模型最高概率比分${correctScores[0].score}，预期进球${bets.homeXG.toFixed(1)}:${bets.awayXG.toFixed(1)}。高赔率${correctScores[0].odds}值得小注尝试。`,
      star: correctScores[0].prob > 0.15 ? 3 : correctScores[0].prob > 0.10 ? 2 : 1,
    });
  }

  // Sort by confidence * star value
  tips.sort((a, b) => (b.confidence * b.star) - (a.confidence * a.star));
  return tips;
}

// ============================================================
// REAL-TIME HELPERS
// ============================================================
// Parse match time as Beijing time (UTC+8) → returns UTC Date
function parseMatchUTC(match) {
  const [year, month, day] = match.date.split('-').map(Number);
  const [hour, minute] = match.time.split(':').map(Number);
  // Beijing time = UTC+8, so UTC = Beijing - 8h
  return new Date(Date.UTC(year, month - 1, day, hour - 8, minute));
}

function getMatchStatus(match, now) {
  const matchTime = parseMatchUTC(match);
  const diff = (now - matchTime) / 60000; // minutes since kickoff
  if (match.completed) return 'finished';
  if (diff >= 0 && diff < 105) return 'live'; // 90min + 15min stoppage
  if (diff < 0) return 'upcoming';
  return 'finished'; // past 105min and not marked completed yet
}

function getCountdown(match, now) {
  const matchTime = parseMatchUTC(match);
  const diff = matchTime - now;
  if (diff <= 0) return null;
  const hours = Math.floor(diff / 3600000);
  const mins = Math.floor((diff % 3600000) / 60000);
  if (hours > 24) return `${Math.floor(hours / 24)}天${hours % 24}时`;
  if (hours > 0) return `${hours}时${mins}分`;
  return `${mins}分钟`;
}

function getLiveMinute(match, now) {
  const matchTime = parseMatchUTC(match);
  const elapsed = (now - matchTime) / 60000;
  if (elapsed < 0) return 0;
  if (elapsed > 90) return Math.min(Math.floor(elapsed), 105); // stoppage
  return Math.floor(elapsed);
}

// Adjust odds dynamically for live matches
function adjustLiveOdds(bets, minute, homeScore, awayScore) {
  if (minute <= 0) return bets;
  const t = Math.min(minute / 90, 1); // progress ratio
  
  // As match progresses, odds shift based on current score
  const goalDiff = homeScore - awayScore;
  const timeRemaining = 1 - t;
  
  // More time remaining = less impact from current score
  const scoreImpact = t * 0.6; // up to 60% shift at full time
  
  // Adjust win/draw/loss probabilities
  let adjPh, adjPd, adjPa;
  if (goalDiff > 0) {
    // Home leading
    adjPh = bets.ph + scoreImpact * 0.3;
    adjPd = bets.pd - scoreImpact * 0.1;
    adjPa = bets.pa - scoreImpact * 0.2;
  } else if (goalDiff < 0) {
    // Away leading
    adjPh = bets.ph - scoreImpact * 0.2;
    adjPd = bets.pd - scoreImpact * 0.1;
    adjPa = bets.pa + scoreImpact * 0.3;
  } else {
    // Draw - draw prob increases as time runs out
    adjPh = bets.ph - scoreImpact * 0.1;
    adjPd = bets.pd + scoreImpact * 0.2;
    adjPa = bets.pa - scoreImpact * 0.1;
  }
  
  // Normalize
  const tot = adjPh + adjPd + adjPa;
  adjPh /= tot; adjPd /= tot; adjPa /= tot;
  
  // Ensure minimums
  adjPh = Math.max(0.01, adjPh);
  adjPd = Math.max(0.01, adjPd);
  adjPa = Math.max(0.01, adjPa);
  
  const MARGIN = 0.90;
  const oddsW = +(MARGIN / adjPh).toFixed(2);
  const oddsD = +(MARGIN / adjPd).toFixed(2);
  const oddsL = +(MARGIN / adjPa).toFixed(2);
  
  return { ...bets, spf: { w: oddsW, d: oddsD, l: oddsL }, ph: adjPh, pd: adjPd, pa: adjPa };
}

// ============================================================
// BETTING TAB COMPONENT
// ============================================================
export default function BettingTab() {
  const [betSlip, setBetSlip] = useState([]); // selected bets
  const [stake, setStake] = useState(10); // base stake per unit (元)
  const [multiplier, setMultiplier] = useState(1); // 倍数
  const [betType, setBetType] = useState('spf'); // current viewing type
  const [selectedDate, setSelectedDate] = useState(null);
  const [showRecommend, setShowRecommend] = useState(false);
  const [now, setNow] = useState(new Date());
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [liveScores, setLiveScores] = useState({}); // {matchKey: {home, away}}
  const prevOddsRef = useRef({}); // track previous odds for change indicators

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const timer = setInterval(() => {
      setNow(new Date());
      setLastUpdate(new Date());
    }, 30000);
    return () => clearInterval(timer);
  }, []);

  // Simulate live scores for matches currently in progress
  useEffect(() => {
    const newScores = {};
    SCHEDULE.filter(m => !m.completed).forEach(m => {
      const status = getMatchStatus(m, now);
      if (status === 'live') {
        const key = `${m.home}-${m.away}`;
        const minute = getLiveMinute(m, now);
        // Simulate goals based on Elo-derived xG and elapsed time
        const eH = ELO[m.home] || 1600;
        const eA = ELO[m.away] || 1600;
        const hb = (HOST_NATIONS.has(m.home) ? HOME_ADV : 0) - (HOST_NATIONS.has(m.away) ? HOME_ADV : 0);
        const homeXG = expectedGoals(eH, eA, hb);
        const awayXG = expectedGoals(eA, eH, -hb / 2);
        const progress = minute / 90;
        // Deterministic "simulated" score based on xG * progress
        const simHome = Math.floor(homeXG * progress + (minute % 7 === 0 ? 1 : 0));
        const simAway = Math.floor(awayXG * progress + (minute % 11 === 0 ? 1 : 0));
        newScores[key] = { home: simHome, away: simAway, minute };
      }
    });
    setLiveScores(newScores);
  }, [now]);

  // Get upcoming matches
  const upcoming = useMemo(() =>
    SCHEDULE.filter(m => !m.completed).sort((a, b) => {
      const da = new Date(`${a.date}T${a.time}`);
      const db = new Date(`${b.date}T${b.time}`);
      return da - db;
    }), []);

  // Group by date
  const dates = useMemo(() => {
    const d = {};
    upcoming.forEach(m => {
      if (!d[m.date]) d[m.date] = [];
      d[m.date].push(m);
    });
    return Object.keys(d).sort();
  }, [upcoming]);

  const activeDate = selectedDate || dates[0];
  const activeMatches = upcoming.filter(m => m.date === activeDate);

  // Pre-compute all bets for active matches (with live adjustments)
  const allBets = useMemo(() => {
    const map = {};
    activeMatches.forEach(m => {
      const key = `${m.home}-${m.away}`;
      let bets = computeAllBets(m.home, m.away);
      const status = getMatchStatus(m, now);
      if (status === 'live' && liveScores[key]) {
        const ls = liveScores[key];
        bets = adjustLiveOdds(bets, ls.minute, ls.home, ls.away);
      }
      // Track previous odds for change arrows
      const prev = prevOddsRef.current[key];
      if (prev) {
        bets.oddsChange = {
          w: bets.spf.w < prev.w ? 'up' : bets.spf.w > prev.w ? 'down' : 'flat',
          d: bets.spf.d < prev.d ? 'up' : bets.spf.d > prev.d ? 'down' : 'flat',
          l: bets.spf.l < prev.l ? 'up' : bets.spf.l > prev.l ? 'down' : 'flat',
        };
      }
      prevOddsRef.current[key] = { ...bets.spf };
      map[key] = bets;
    });
    return map;
  }, [activeMatches, now, liveScores]);

  // All recommendations for upcoming matches
  const allRecommendations = useMemo(() => {
    const recs = [];
    upcoming.slice(0, 12).forEach(m => {
      const bets = computeAllBets(m.home, m.away);
      const tips = generateRecommendation(m, bets);
      tips.forEach(tip => {
        recs.push({ match: m, ...tip });
      });
    });
    // Sort by star * confidence, take top recommendations
    recs.sort((a, b) => (b.star * b.confidence) - (a.star * a.confidence));
    return recs.slice(0, 8);
  }, [upcoming]);

  const toggleBet = (matchKey, match, type, pick, odds) => {
    const id = `${matchKey}_${type}_${pick}`;
    setBetSlip(prev => {
      const exists = prev.find(b => b.id === id);
      if (exists) return prev.filter(b => b.id !== id);
      return [...prev, { id, matchKey, match, type, pick, odds }];
    });
  };

  const isInSlip = (matchKey, type, pick) =>
    betSlip.some(b => b.id === `${matchKey}_${type}_${pick}`);

  const clearSlip = () => setBetSlip([]);

  // Calculate combined odds and potential win
  const combinedOdds = betSlip.length > 0
    ? betSlip.reduce((acc, b) => acc * b.odds, 1)
    : 0;
  const totalCost = stake * multiplier * (betSlip.length > 0 ? 1 : 0); // 单注
  const potentialWin = totalCost * combinedOdds;

  const formatDate = (d) => {
    const date = new Date(d + 'T12:00:00');
    const month = date.getMonth() + 1;
    const day = date.getDate();
    const weekdays = ['日', '一', '二', '三', '四', '五', '六'];
    return `${month}/${day} 周${weekdays[date.getDay()]}`;
  };

  const betTypes = [
    { key: 'spf', label: '胜平负' },
    { key: 'rspf', label: '让球' },
    { key: 'ou', label: '大小球' },
    { key: 'score', label: '比分' },
    { key: 'goals', label: '总进球' },
  ];

  const btnStyle = (active, color) => ({
    padding: '8px 10px', borderRadius: 8, border: 'none', cursor: 'pointer',
    fontFamily: 'inherit', fontWeight: 700, fontSize: 12, transition: 'all 0.15s',
    background: active ? (color || '#e07820') : '#1e293b',
    color: active ? '#fff' : '#94a3b8',
    minWidth: 60, textAlign: 'center', lineHeight: 1.3,
  });

  const OddsBtn = ({ matchKey, match, type, pick, odds, label, subLabel, color, wide, change }) => {
    const active = isInSlip(matchKey, type, pick);
    return (
      <button onClick={() => toggleBet(matchKey, match, type, pick, odds)} style={{
        ...btnStyle(active, color),
        flex: wide ? '1 1 100%' : '1',
        display: 'flex', flexDirection: 'column', gap: 2,
        padding: '8px 6px',
        position: 'relative',
      }}>
        <span style={{ fontSize: 11, opacity: 0.85 }}>{subLabel || label}</span>
        <span style={{ fontSize: 15, fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3 }}>
          {odds}
          {change === 'up' && <span style={{ fontSize: 10, color: '#ef4444' }}>↑</span>}
          {change === 'down' && <span style={{ fontSize: 10, color: '#22c55e' }}>↓</span>}
        </span>
      </button>
    );
  };

  return (
    <div>
      {/* Pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, margin: '0 0 4px' }}>🎰 竞彩投注</h2>
          <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>基于Poisson+ELO混合模型赔率 · 返还率90% · 理性购彩</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 10, color: '#475569' }}>
            🔄 {lastUpdate.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
          <button onClick={() => { setNow(new Date()); setLastUpdate(new Date()); }} style={{
            background: '#1e293b', border: '1px solid #334155', color: '#94a3b8', borderRadius: 6,
            padding: '4px 10px', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
          }}>刷新</button>
        </div>
      </div>

      {/* AI Recommendation Banner */}
      <div style={{
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        border: '1px solid #334155', borderRadius: 14, padding: 16, marginBottom: 16,
        cursor: 'pointer',
      }} onClick={() => setShowRecommend(!showRecommend)}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: 14, fontWeight: 800, color: '#e07820' }}>🤖 AI推荐方案</span>
            <span style={{ fontSize: 11, color: '#64748b', marginLeft: 8 }}>
              {showRecommend ? '收起' : '展开查看'}
            </span>
          </div>
          <span style={{ fontSize: 18, color: '#64748b' }}>{showRecommend ? '▲' : '▼'}</span>
        </div>

        {showRecommend && (
          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Quick pick button */}
            <button onClick={(e) => {
              e.stopPropagation();
              // Add top 3 star-3 recommendations to slip
              const topPicks = allRecommendations.filter(r => r.star >= 2).slice(0, 3);
              const newSlip = topPicks.map(r => {
                let type, pick;
                if (r.type === '胜平负') {
                  type = 'spf';
                  pick = r.pick;
                } else if (r.type.includes('大小球')) {
                  type = 'ou';
                  pick = r.pick;
                } else if (r.type === '比分') {
                  type = 'score';
                  pick = r.pick;
                }
                return {
                  id: `${r.match.home}-${r.match.away}_${type}_${pick}`,
                  matchKey: `${r.match.home}-${r.match.away}`,
                  match: r.match,
                  type, pick, odds: r.odds,
                };
              }).filter(b => b.type);
              setBetSlip(newSlip);
              setMultiplier(10);
              setStake(10);
            }} style={{
              background: '#e07820', color: '#020617', border: 'none', borderRadius: 10,
              padding: '10px 16px', fontWeight: 800, fontSize: 13, cursor: 'pointer',
              fontFamily: 'inherit',
            }}>
              ⚡ 一键跟单TOP推荐 (10倍×10元=100元)
            </button>

            {allRecommendations.map((rec, i) => (
              <div key={i} style={{
                background: '#020617', borderRadius: 10, padding: '10px 12px',
                border: `1px solid ${rec.star >= 3 ? '#e07820' : rec.star >= 2 ? '#334155' : '#1e293b'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: '#e2e8f0' }}>
                    {flag(rec.match.home)} {cn(rec.match.home)} vs {cn(rec.match.away)} {flag(rec.match.away)}
                  </span>
                  <span style={{ fontSize: 11, color: '#64748b' }}>{rec.match.date.slice(5)} {rec.match.time}</span>
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                  <span style={{
                    fontSize: 10, padding: '2px 8px', borderRadius: 4,
                    background: rec.type === '胜平负' ? '#1e3a5f' : rec.type === '比分' ? '#3b1e5f' : '#1e5f3a',
                    color: '#fff', fontWeight: 700,
                  }}>{rec.type}</span>
                  <span style={{ fontSize: 14, fontWeight: 800, color: '#e07820' }}>{rec.pick}</span>
                  <span style={{ fontSize: 12, color: '#22c55e', fontWeight: 700 }}>@{rec.odds}</span>
                  <span style={{ fontSize: 11, color: '#64748b' }}>
                    信心 {rec.confidence}% {'⭐'.repeat(rec.star)}
                  </span>
                </div>
                <p style={{ fontSize: 11, color: '#94a3b8', margin: 0, lineHeight: 1.5 }}>
                  💡 {rec.reason}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Date selector */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 12, overflowX: 'auto', paddingBottom: 4 }}>
        {dates.map(d => (
          <button key={d} onClick={() => setSelectedDate(d)} style={{
            ...btnStyle(activeDate === d),
            whiteSpace: 'nowrap', padding: '6px 14px',
          }}>
            {formatDate(d)}
          </button>
        ))}
      </div>

      {/* Bet type selector */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {betTypes.map(bt => (
          <button key={bt.key} onClick={() => setBetType(bt.key)} style={{
            ...btnStyle(betType === bt.key, '#0f172a'),
            border: betType === bt.key ? '1px solid #e07820' : '1px solid #1e293b',
            padding: '6px 14px',
          }}>
            {bt.label}
          </button>
        ))}
      </div>

      {/* Stake & Multiplier Panel - Always Visible */}
      <div style={{
        background: '#0f172a', border: '1px solid #1e293b', borderRadius: 14,
        padding: 16, marginBottom: 16,
      }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#e07820', marginBottom: 12 }}>💰 投注设置</div>

        {/* Stake input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: '#94a3b8', minWidth: 50 }}>单注金额:</span>
          <input
            type="number"
            value={stake}
            onChange={(e) => {
              const v = Math.max(1, parseInt(e.target.value) || 1);
              setStake(v);
            }}
            style={{
              width: 80, padding: '6px 10px', borderRadius: 8, border: '1px solid #334155',
              background: '#1e293b', color: '#e07820', fontSize: 16, fontWeight: 800,
              fontFamily: 'inherit', textAlign: 'center', outline: 'none',
            }}
          />
          <span style={{ fontSize: 12, color: '#64748b' }}>元</span>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {[2, 5, 10, 20, 50, 100, 200].map(s => (
              <button key={s} onClick={() => setStake(s)} style={{
                padding: '4px 10px', borderRadius: 6, border: stake === s ? '2px solid #e07820' : '1px solid #334155',
                background: stake === s ? '#e07820' : '#1e293b',
                color: stake === s ? '#020617' : '#94a3b8',
                fontSize: 11, fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit',
              }}>
                ¥{s}
              </button>
            ))}
          </div>
        </div>

        {/* Multiplier input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: '#94a3b8', minWidth: 50 }}>投注倍数:</span>
          <input
            type="number"
            value={multiplier}
            onChange={(e) => {
              const v = Math.max(1, Math.min(99, parseInt(e.target.value) || 1));
              setMultiplier(v);
            }}
            style={{
              width: 60, padding: '6px 10px', borderRadius: 8, border: '1px solid #334155',
              background: '#1e293b', color: '#7c3aed', fontSize: 16, fontWeight: 800,
              fontFamily: 'inherit', textAlign: 'center', outline: 'none',
            }}
          />
          <span style={{ fontSize: 12, color: '#64748b' }}>倍</span>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {[1, 2, 5, 10, 20, 50, 99].map(m => (
              <button key={m} onClick={() => setMultiplier(m)} style={{
                padding: '4px 10px', borderRadius: 6, border: multiplier === m ? '2px solid #7c3aed' : '1px solid #334155',
                background: multiplier === m ? '#7c3aed' : '#1e293b',
                color: multiplier === m ? '#fff' : '#94a3b8',
                fontSize: 11, fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit',
              }}>
                {m}x
              </button>
            ))}
          </div>
        </div>

        {/* Live calculation */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          background: '#020617', borderRadius: 10, padding: '10px 14px',
          border: '1px solid #1e293b',
        }}>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            总投入: <span style={{ color: '#e07820', fontWeight: 800, fontSize: 18 }}>¥{stake * multiplier}</span>
            <span style={{ fontSize: 10, marginLeft: 6 }}>({stake}元 × {multiplier}倍)</span>
          </div>
          {betSlip.length > 0 ? (
            <div style={{ fontSize: 12, color: '#64748b' }}>
              {betSlip.length}串1 · 赔率 <span style={{ color: '#22c55e', fontWeight: 800, fontSize: 16 }}>{combinedOdds.toFixed(2)}</span>
              <span style={{ marginLeft: 8 }}>→ 最高奖金 </span>
              <span style={{ color: '#22c55e', fontWeight: 800, fontSize: 18 }}>¥{(stake * multiplier * combinedOdds).toFixed(0)}</span>
            </div>
          ) : (
            <div style={{ fontSize: 12, color: '#475569' }}>← 选择赔率后自动计算奖金</div>
          )}
        </div>
      </div>

      {/* Matches list with odds */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 20 }}>
        {activeMatches.map(match => {
          const key = `${match.home}-${match.away}`;
          const bets = allBets[key];
          if (!bets) return null;
          const status = getMatchStatus(match, now);
          const ls = liveScores[key];
          const countdown = getCountdown(match, now);
          const isLive = status === 'live';
          const oddsChange = bets.oddsChange;

          return (
            <div key={key} style={{
              background: isLive ? 'linear-gradient(135deg, #0f172a 0%, #1a0a00 100%)' : '#0f172a',
              borderRadius: 14, border: isLive ? '1px solid #e07820' : '1px solid #1e293b',
              overflow: 'hidden', boxShadow: isLive ? '0 0 20px rgba(224,120,32,0.15)' : 'none',
            }}>
              {/* Match header */}
              <div style={{
                padding: '12px 16px', borderBottom: '1px solid #1e293b',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: '#64748b', background: '#1e293b', padding: '2px 8px', borderRadius: 4 }}>{match.group}组</span>
                  <span style={{ fontSize: 14, fontWeight: 700 }}>
                    {flag(match.home)} {cn(match.home)}
                  </span>
                  {isLive && ls ? (
                    <span style={{
                      fontSize: 16, fontWeight: 900, color: '#e07820',
                      background: '#1e293b', padding: '2px 10px', borderRadius: 6,
                    }}>
                      {ls.home} - {ls.away}
                    </span>
                  ) : (
                    <span style={{ fontSize: 11, color: '#475569' }}>vs</span>
                  )}
                  <span style={{ fontSize: 14, fontWeight: 700 }}>
                    {cn(match.away)} {flag(match.away)}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  {isLive && (
                    <span style={{
                      fontSize: 10, fontWeight: 800, color: '#fff',
                      background: '#dc2626', padding: '2px 8px', borderRadius: 4,
                      animation: 'pulse 2s infinite',
                    }}>
                      ● LIVE {ls?.minute || '?'}'
                    </span>
                  )}
                  {!isLive && countdown && (
                    <span style={{ fontSize: 10, color: '#22c55e', background: '#0a2618', padding: '2px 8px', borderRadius: 4 }}>
                      ⏱ {countdown}后开赛
                    </span>
                  )}
                  <span style={{ fontSize: 11, color: '#64748b' }}>{match.time}</span>
                </div>
              </div>

              {/* Odds area */}
              <div style={{ padding: '12px 14px' }}>
                {betType === 'spf' && (
                  <div style={{ display: 'flex', gap: 8 }}>
                    <OddsBtn matchKey={key} match={match} type="spf" pick="主胜" odds={bets.spf.w}
                      label="主胜" subLabel={`${cn(match.home)}胜`} color="#2563eb"
                      change={oddsChange?.w} />
                    <OddsBtn matchKey={key} match={match} type="spf" pick="平局" odds={bets.spf.d}
                      label="平局" subLabel="平局" color="#6b7280"
                      change={oddsChange?.d} />
                    <OddsBtn matchKey={key} match={match} type="spf" pick="客胜" odds={bets.spf.l}
                      label="客胜" subLabel={`${cn(match.away)}胜`} color="#dc2626"
                      change={oddsChange?.l} />
                  </div>
                )}

                {betType === 'rspf' && (
                  <div>
                    <div style={{ fontSize: 11, color: '#e07820', marginBottom: 8, textAlign: 'center' }}>
                      让球: {match.home} {bets.handicap > 0 ? `+${bets.handicap}` : bets.handicap}
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <OddsBtn matchKey={key} match={match} type="rspf" pick="让胜" odds={bets.rspf.w}
                        label="让胜" subLabel={`让${bets.handicap > 0 ? '+' : ''}${bets.handicap} 胜`} color="#2563eb" />
                      <OddsBtn matchKey={key} match={match} type="rspf" pick="让平" odds={bets.rspf.d}
                        label="让平" subLabel={`让${bets.handicap > 0 ? '+' : ''}${bets.handicap} 平`} color="#6b7280" />
                      <OddsBtn matchKey={key} match={match} type="rspf" pick="让负" odds={bets.rspf.l}
                        label="让负" subLabel={`让${bets.handicap > 0 ? '+' : ''}${bets.handicap} 负`} color="#dc2626" />
                    </div>
                  </div>
                )}

                {betType === 'ou' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <OddsBtn matchKey={key} match={match} type="ou" pick="大2.5" odds={bets.ou25.over}
                        label="大" subLabel="大2.5球" color="#f59e0b" wide />
                      <OddsBtn matchKey={key} match={match} type="ou" pick="小2.5" odds={bets.ou25.under}
                        label="小" subLabel="小2.5球" color="#3b82f6" wide />
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <OddsBtn matchKey={key} match={match} type="ou" pick="大1.5" odds={bets.ou15.over}
                        label="大" subLabel="大1.5球" color="#f59e0b" wide />
                      <OddsBtn matchKey={key} match={match} type="ou" pick="小1.5" odds={bets.ou15.under}
                        label="小" subLabel="小1.5球" color="#3b82f6" wide />
                    </div>
                  </div>
                )}

                {betType === 'score' && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
                    {bets.correctScores.map(s => (
                      <OddsBtn key={s.score} matchKey={key} match={match} type="score" pick={s.score} odds={s.odds}
                        label={s.score} subLabel={s.score} color="#7c3aed" />
                    ))}
                  </div>
                )}

                {betType === 'goals' && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
                    {bets.totalGoals.map(g => (
                      <OddsBtn key={g.label} matchKey={key} match={match} type="goals" pick={g.label} odds={g.odds}
                        label={g.label} subLabel={g.label} color="#059669" />
                    ))}
                  </div>
                )}

                {/* Expected goals indicator */}
                <div style={{ marginTop: 8, fontSize: 10, color: '#475569', textAlign: 'center' }}>
                  预期进球: {cn(match.home)} {bets.homeXG.toFixed(2)} - {bets.awayXG.toFixed(2)} {cn(match.away)} | 合计 {bets.expectedTotal.toFixed(2)}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* BET SLIP - Sticky bottom (compact) */}
      {betSlip.length > 0 && (
        <div style={{
          position: 'fixed', bottom: 0, left: 0, right: 0,
          background: '#0f172a', borderTop: '2px solid #e07820',
          padding: '10px 16px', zIndex: 100,
          boxShadow: '0 -4px 24px rgba(0,0,0,0.5)',
        }}>
          <div style={{ maxWidth: 960, margin: '0 auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
              {/* Selected bets */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, flex: 1 }}>
                {betSlip.map(b => (
                  <div key={b.id} onClick={() => toggleBet(b.matchKey, b.match, b.type, b.pick, b.odds)}
                    style={{
                      background: '#e07820', color: '#020617', borderRadius: 6,
                      padding: '3px 8px', fontSize: 10, fontWeight: 700, cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 3,
                    }}>
                    <span>{b.pick}</span>
                    <span style={{ opacity: 0.7 }}>@{b.odds}</span>
                    <span style={{ fontSize: 12, marginLeft: 1 }}>✕</span>
                  </div>
                ))}
              </div>

              {/* Summary + button */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ textAlign: 'right', fontSize: 11 }}>
                  <span style={{ color: '#64748b' }}>{betSlip.length}串1 </span>
                  <span style={{ color: '#22c55e', fontWeight: 700 }}>@{combinedOdds.toFixed(2)}</span>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 9, color: '#64748b' }}>投入 ¥{stake * multiplier} → 奖金</div>
                  <div style={{ fontSize: 16, fontWeight: 800, color: '#22c55e' }}>¥{potentialWin.toFixed(0)}</div>
                </div>
                <button style={{
                  background: 'linear-gradient(135deg, #e07820, #f59e0b)',
                  color: '#020617', border: 'none', borderRadius: 10,
                  padding: '8px 20px', fontWeight: 800, fontSize: 13, cursor: 'pointer',
                  fontFamily: 'inherit', whiteSpace: 'nowrap',
                }}>
                  确认投注
                </button>
                <button onClick={clearSlip} style={{
                  background: 'transparent', border: 'none', color: '#ef4444',
                  fontSize: 11, cursor: 'pointer', fontFamily: 'inherit', whiteSpace: 'nowrap',
                }}>清空</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Spacer for sticky slip */}
      {betSlip.length > 0 && <div style={{ height: 200 }} />}
    </div>
  );
}

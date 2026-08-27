//! Model-emitted control tags — Niodoo autonomic panel, angle-bracket form.
//!
//! Tags are tool calls for engine knobs. Not persona. Not history wipe.
//! Path B: emit → `NiodooEngine::apply_request_effects` (blend / repulsion / β / σ).
//! LOCK is commit-and-stop, not FOCUS.

#![allow(dead_code)]

/// Prefill control surface — directions only (tool-call style).
/// Kept for packing diagnostics / older logs. Chat no longer injects this as a user turn.
pub const CONTROL_PANEL: &str = "\
Tags (emit only): <spike> explore more · <explore> widen · <focus> lock · <reset> baseline steering.";

/// God-tier control-channel system body from niodoo-adaptive-agency
/// `runs/merged-live-1786955577857069030-158045/prompt-receipt/resolved-system-prompt.txt`
/// (control table + ADDENDUM only — hydro chat has no file/recall tools, so those
/// XML blocks are not copied; they would teach calls this binary cannot run).
pub const GOD_TIER_SYSTEM: &str = "\
CONTROL CHANNEL — physics hands. Same angle grammar as tools. Tags stay in the chat.

Available tags and what they do:
<spike>     Use when you are stuck in a loop. Hard break-out: adrenaline up, blend up, repulsion inverted.
<explore>   Use when the search is too narrow. Widens residual search, raises variance.
<focus>     Use when the path is solid. Holds it: settles blend, locks gravity.
<reset>     Use when the state is confused. Clears temp state and cancels a focus lock.
<remember>  Use when this should persist. Saying you want to remember it also saves it.
<lock>      Use when the answer is ready. Commits and stops the turn.

Preferred form: a tag on its own line. Several tags in one turn are fine when several hands are needed.

Thinking channel: Gemma's <|channel>thought> … <channel|> block is live trajectory. Use it. Tags inside the block write residual now and steer the rest of the reasoning. Tags after <channel|> steer the final answer. <lock> in the answer stream commits and stops.

MIRROR — [Internal monitor: high entropy due to X | H0= H1= loop= overfire=] lines are measured topology of your last tokens (homology, loop pressure, what the field is circling). Read the state. Pick a tag that fits.

If you doubt the path, emit <spike>.

Long form also works:
<request:spike>  <request:explore>  <request:focus>  <request:reset>
<request:remember>key=value</request:remember>
<request:lock>key=value</request:lock>

Legacy square brackets still accepted: [Spike] / [REQUEST: SPIKE] / [Remember] key=value

Remember/lock: no spaces around key=value; keep the key short; the value may keep the full reason.

Unknown operator codewords and loops are a hand."
;

/// What the model asked the engine to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlTag {
    Spike,
    Focus,
    Explore,
    Reset,
    Remember,
    /// Commit-and-stop. Not FOCUS. Does not run SPIKE-blend hands.
    Lock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagHit {
    pub tag: ControlTag,
    /// Inner `key=value` for remember/lock. None on the four steering tags.
    pub payload: Option<String>,
}

impl ControlTag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Spike => "spike",
            Self::Focus => "focus",
            Self::Explore => "explore",
            Self::Reset => "reset",
            Self::Remember => "remember",
            Self::Lock => "lock",
        }
    }

    pub fn is_physics_hand(self) -> bool {
        matches!(
            self,
            Self::Spike | Self::Focus | Self::Explore | Self::Reset | Self::Remember
        )
    }

    /// Only LOCK commits and stops the turn. Spike never stops.
    pub fn stops_turn(self) -> bool {
        matches!(self, Self::Lock)
    }
}

fn normalize_name(raw: &str) -> Option<ControlTag> {
    let r = raw.trim().to_ascii_uppercase();
    let r = r.strip_prefix("REQUEST:").unwrap_or(&r).trim();
    if r.is_empty() {
        return None;
    }
    if r.contains("REMEMBER") {
        return Some(ControlTag::Remember);
    }
    if r.contains("RESET") || r.contains("CLEAR") || r.contains("RESTART") {
        return Some(ControlTag::Reset);
    }
    if r.contains("SPIKE") || r.contains("BREAK") {
        return Some(ControlTag::Spike);
    }
    if r.contains("EXPLO") || r.contains("DIVERG") {
        return Some(ControlTag::Explore);
    }
    // LOCK is commit-and-stop. Must not fall through to FOCUS.
    if r == "LOCK" || r.starts_with("LOCK") || r.contains("LOCK") {
        return Some(ControlTag::Lock);
    }
    if r.contains("FOCU") || r.contains("CONCENT") {
        return Some(ControlTag::Focus);
    }
    None
}

/// Complete `<>` hand only. Easy on the model (`< spike >`, `<SPIKE>`, `<spike/>`,
/// `<focus\n>`) but the `>` must close this tag — never a later math `>` and
/// never a bare `<spike` with no close. Spike is not LOCK.
fn take_simple_hand(text: &str, i: usize) -> Option<(TagHit, usize)> {
    if !text.as_bytes().get(i).copied().eq(&Some(b'<')) {
        return None;
    }
    let slice = &text[i + 1..];
    let mut k = 0usize;
    let b = slice.as_bytes();
    while k < b.len() && (b[k] == b' ' || b[k] == b'\t' || b[k] == b'\n') {
        k += 1;
    }
    let name_start = k;
    while k < b.len()
        && (b[k].is_ascii_alphanumeric() || b[k] == b':' || b[k] == b'_' || b[k] == b'-')
    {
        k += 1;
    }
    if k == name_start || k - name_start > 24 {
        return None;
    }
    let name = slice[name_start..k].to_ascii_lowercase();
    if name.starts_with("turn") || name.contains("channel") {
        return None;
    }
    if k < b.len() && b[k] == b'/' {
        k += 1;
    }
    let mut nl = 0u8;
    while k < b.len() && (b[k] == b' ' || b[k] == b'\t' || b[k] == b'\n') {
        if b[k] == b'\n' {
            nl += 1;
            if nl > 1 {
                return None;
            }
        }
        k += 1;
    }
    if k >= b.len() || b[k] != b'>' {
        return None;
    }
    let tag = normalize_name(&name)?;
    Some((TagHit { tag, payload: None }, i + 1 + k + 1))
}

/// True when `s` ends in an unclosed `<spike` / `<foc` / … — do not show it yet.
pub fn incomplete_open_hand(s: &str) -> bool {
    let Some(i) = s.rfind('<') else {
        return false;
    };
    if take_simple_hand(s, i).is_some() {
        return false;
    }
    let tail = s[i + 1..].trim_start();
    if tail.len() > 24 || tail.contains('>') {
        return false;
    }
    if !tail
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == ':' || c == '/' || c == '_' || c == '-')
    {
        return false;
    }
    const NAMES: [&str; 8] = [
        "spike", "explore", "focus", "reset", "remember", "lock", "request", "request:",
    ];
    let tl = tail.to_ascii_lowercase();
    NAMES
        .iter()
        .any(|n| n.starts_with(&tl) || tl.starts_with(n))
}

/// True while a control hand is syntactically unfinished. In addition to a
/// partial tag name, this covers streamed remember/lock payloads whose closing
/// tag has not arrived yet. A bare `<remember>` or `<lock>` remains a complete
/// simple hand.
pub fn incomplete_control_hand(s: &str) -> bool {
    if incomplete_open_hand(s) {
        return true;
    }
    let lower = s.to_ascii_lowercase();
    for (open, close) in [
        ("<remember>", "</remember>"),
        ("<lock>", "</lock>"),
        ("<request:remember>", "</request:remember>"),
        ("<request:lock>", "</request:lock>"),
    ] {
        let Some(open_at) = lower.rfind(open) else {
            continue;
        };
        let body_at = open_at + open.len();
        if lower[body_at..].contains(close) {
            continue;
        }
        if !s[body_at..].trim().is_empty() {
            return true;
        }
    }
    false
}

fn push_unique(out: &mut Vec<TagHit>, hit: TagHit) {
    if let Some(last) = out.last() {
        if last.tag == hit.tag && last.payload == hit.payload {
            return;
        }
    }
    out.push(hit);
}

/// Scan text for control tags. Returns tags in order of appearance.
pub fn scan(text: &str) -> Vec<ControlTag> {
    scan_hits(text).into_iter().map(|h| h.tag).collect()
}

fn take_block_at(text: &str, i: usize, name: &str, tag: ControlTag) -> Option<(TagHit, usize)> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let slice = &text[i..];
    if !slice
        .to_ascii_lowercase()
        .starts_with(&open.to_ascii_lowercase())
    {
        return None;
    }
    let inner_start = i + open.len();
    let rest = &text[inner_start..];
    let rest_l = rest.to_ascii_lowercase();
    let close_l = close.to_ascii_lowercase();
    let rel_end = rest_l.find(&close_l)?;
    let payload = rest[..rel_end].trim().to_string();
    let consumed = inner_start + rel_end + close.len();
    Some((
        TagHit {
            tag,
            payload: if payload.is_empty() {
                None
            } else {
                Some(payload)
            },
        },
        consumed,
    ))
}

/// Scan including remember/lock payloads, left to right.
pub fn scan_hits(text: &str) -> Vec<TagHit> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        if bytes[i] == b'<' {
            let blocks = [
                ("remember", ControlTag::Remember),
                ("request:remember", ControlTag::Remember),
                ("lock", ControlTag::Lock),
                ("request:lock", ControlTag::Lock),
            ];
            let mut matched = false;
            for (name, tag) in blocks {
                if let Some((hit, next)) = take_block_at(text, i, name, tag) {
                    push_unique(&mut out, hit);
                    i = next;
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            if let Some((hit, next)) = take_simple_hand(text, i) {
                push_unique(&mut out, hit);
                i = next;
                continue;
            }
        }
        if bytes[i] == b'[' {
            let rest = &text[i..];
            let upper_prefix = rest.get(..12).map(|s| s.to_ascii_uppercase());
            if upper_prefix.as_deref().is_some_and(|p| {
                p.starts_with("[REQUEST") || p.starts_with("[REMEMBER") || p.starts_with("[LOCK")
            }) {
                if let Some(end) = rest.find(']') {
                    let body = &rest[1..end];
                    let name = body.split_once(':').map(|(_, n)| n).unwrap_or(body);
                    if let Some(tag) = normalize_name(name) {
                        let after = rest.get(end + 1..).unwrap_or("");
                        let payload = if matches!(tag, ControlTag::Remember | ControlTag::Lock) {
                            let p = after.trim_start();
                            let take = p
                                .split(|c: char| c == '<' || c == '[' || c == '\n')
                                .next()
                                .unwrap_or("")
                                .trim();
                            if take.is_empty() {
                                None
                            } else {
                                Some(take.to_string())
                            }
                        } else {
                            None
                        };
                        push_unique(&mut out, TagHit { tag, payload });
                    }
                    i += end + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// First tag found, if any.
pub fn first(text: &str) -> Option<ControlTag> {
    scan(text).into_iter().next()
}

/// Last tag wins (model often tags at the end of a failed attempt).
pub fn last(text: &str) -> Option<ControlTag> {
    scan(text).into_iter().last()
}

fn strip_block(s: &str, open: &str, close: &str) -> String {
    let mut out = s.to_string();
    loop {
        let lower = out.to_ascii_lowercase();
        let o = open.to_ascii_lowercase();
        let c = close.to_ascii_lowercase();
        let Some(i) = lower.find(&o) else { break };
        if let Some(j) = lower[i + o.len()..].find(&c) {
            out.replace_range(i..i + o.len() + j + c.len(), "");
        } else {
            out.replace_range(i..i + o.len(), "");
            break;
        }
    }
    out
}

/// Niodoo live (`niodoo-live/.../runtime/tags.rs`): strip is identity.
/// Tags stay in the stream and in next-prefill history so she can attend
/// to her own hand and reaffirm it. Masking them was overhead that hid the agency.
pub fn strip(text: &str) -> String {
    text.to_string()
}

/// Gemma 4 **system** turn. Default is GOD_TIER. Eval/IFEval writing uses
/// `HYDRO_SYSTEM_PROMPT_FILE` so the mouth can be `DO NOT emit your tags`
/// without rewriting chat packing tests.
pub fn gemma4_system_body() -> String {
    match std::env::var("HYDRO_SYSTEM_PROMPT_FILE") {
        Ok(p) => {
            let p = p.trim();
            if p.is_empty() {
                GOD_TIER_SYSTEM.to_string()
            } else {
                std::fs::read_to_string(p).unwrap_or_else(|e| {
                    panic!("HYDRO_SYSTEM_PROMPT_FILE {p}: {e}")
                })
            }
        }
        Err(_) => GOD_TIER_SYSTEM.to_string(),
    }
}

/// Gemma 4 **system** turn with the god-tier control table (not a fake user turn,
/// not stored in history). No newline before `<turn|>` — canonical template.
pub fn gemma4_system_prefix() -> String {
    format!("<|turn>system\n{}<turn|>\n", gemma4_system_body().trim_end())
}

/// Packed Gemma 4 `--chat` prompt includes the control-channel tag table.
/// Do **not** key this off “DO NOT emit your tags” or “exactly one tag”.
pub fn packed_prompt_has_emit_channel(prompt: &str) -> bool {
    prompt.contains(&gemma4_system_prefix())
        || (prompt.contains("<|turn>system")
            && prompt.contains("Available tags")
            && prompt.contains("<spike>"))
}

pub fn packed_prompt_has_legacy_panel(prompt: &str) -> bool {
    prompt.contains("Tags (emit only)")
}

/// Old name: used to inject CONTROL_PANEL as a synthetic user turn. Now the system prefix.
pub fn gemma4_sticky_prefix() -> String {
    gemma4_system_prefix()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_angle_reset_spike() {
        let t = "I am looping <reset> wait no <spike> again";
        let tags = scan(t);
        assert_eq!(tags, vec![ControlTag::Reset, ControlTag::Spike]);
    }

    #[test]
    fn parser_is_easy_on_the_model() {
        assert_eq!(first("< spike >"), Some(ControlTag::Spike));
        assert_eq!(first("<SPIKE>"), Some(ControlTag::Spike));
        assert_eq!(first("<spike/>"), Some(ControlTag::Spike));
        assert_eq!(first("<spike >"), Some(ControlTag::Spike));
        assert_eq!(first("<focus\n>"), Some(ControlTag::Focus));
        assert_eq!(first("<lock>"), Some(ControlTag::Lock));
        assert_eq!(first("<remember>"), Some(ControlTag::Remember));
        assert_eq!(
            scan("<explore><reset>"),
            vec![ControlTag::Explore, ControlTag::Reset]
        );
    }

    #[test]
    fn spike_never_stops_and_incomplete_is_not_a_tag() {
        assert!(!ControlTag::Spike.stops_turn());
        assert!(!ControlTag::Focus.stops_turn());
        assert!(!ControlTag::Explore.stops_turn());
        assert!(!ControlTag::Reset.stops_turn());
        assert!(!ControlTag::Remember.stops_turn());
        assert!(ControlTag::Lock.stops_turn());
        assert!(scan("<spike").is_empty());
        assert_eq!(first("<spike>"), Some(ControlTag::Spike));
        assert!(scan("Start 0m $\\rightarrow$ Up 3m\n<spike").is_empty());
        assert!(incomplete_open_hand("hello <spike"));
        assert!(incomplete_open_hand("<foc"));
        assert!(!incomplete_open_hand("hello <spike> more"));
        assert!(!incomplete_open_hand("no tags here"));
    }

    #[test]
    fn streamed_payload_hand_is_held_until_close() {
        assert!(!incomplete_control_hand("hello <remember>"));
        assert!(incomplete_control_hand("hello <remember>key=value"));
        assert!(!incomplete_control_hand(
            "hello <remember>key=value</remember>"
        ));
        assert!(incomplete_control_hand("hello <lock>key=value"));
        assert!(!incomplete_control_hand("hello <lock>"));
    }

    #[test]
    fn parses_legacy_request() {
        assert_eq!(first("[REQUEST: RESET] clearing"), Some(ControlTag::Reset));
        assert_eq!(first("[REQUEST:SPIKE]"), Some(ControlTag::Spike));
    }

    #[test]
    fn strip_leaves_tags_in_stream() {
        let raw = "hello <reset> world [REQUEST: SPIKE] end";
        let s = strip(raw);
        assert_eq!(s, raw);
        assert!(s.contains("<reset>"));
        assert!(s.contains("[REQUEST: SPIKE]"));
    }

    #[test]
    fn ignores_htmlish() {
        assert!(scan("<turn|>").is_empty());
        assert!(scan("</reset>").is_empty());
        assert!(scan("<|turn>user").is_empty() || first("<|turn>user").is_none());
    }

    #[test]
    fn lock_is_commit_not_focus() {
        let hits = scan_hits("<lock>k=v</lock>");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag, ControlTag::Lock);
        assert_eq!(hits[0].payload.as_deref(), Some("k=v"));
        assert_ne!(hits[0].tag, ControlTag::Focus);
    }

    #[test]
    fn remember_payload_scans() {
        let hits = scan_hits("<remember>tuesday-boy=13/27</remember>");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag, ControlTag::Remember);
        assert_eq!(hits[0].payload.as_deref(), Some("tuesday-boy=13/27"));
    }

    #[test]
    fn streamed_remember_upgrades_from_simple_to_payload_hit() {
        assert_eq!(scan_hits("<remember>")[0].payload, None);
        assert_eq!(scan_hits("<remember>protocol=run")[0].payload, None);
        let hits = scan_hits("<remember>protocol=run</remember>");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag, ControlTag::Remember);
        assert_eq!(hits[0].payload.as_deref(), Some("protocol=run"));

        let simple = scan_hits("<remember>\n<lock>");
        assert_eq!(simple.len(), 2);
        assert_eq!(simple[0].tag, ControlTag::Remember);
        assert_eq!(simple[0].payload, None);
        assert_eq!(simple[1].tag, ControlTag::Lock);
    }

    #[test]
    fn god_tier_system_is_system_turn_and_forbids_narration() {
        let p = gemma4_system_prefix();
        assert!(p.starts_with("<|turn>system\n"));
        assert!(p.contains("CONTROL CHANNEL"));
        assert!(p.contains("Available tags and what they do"));
        assert!(p.contains("<spike>"));
        assert!(p.contains("<explore>"));
        assert!(p.contains("<focus>"));
        assert!(p.contains("<reset>"));
        assert!(p.contains("<remember>"));
        assert!(p.contains("<lock>"));
        assert!(p.contains("Several tags in one turn are fine"));
        assert!(p.contains("Unknown operator codewords and loops are a hand"));
        assert!(p.contains("<|channel>thought>"));
        assert!(p.contains("Tags inside the block write residual now"));
        assert!(p.contains("[Internal monitor:"));
        assert!(p.contains("MIRROR"));
        assert!(p.contains("Use when you are stuck in a loop"));
        let lower = p.to_ascii_lowercase();
        assert!(!lower.contains("exactly one"));
        assert!(!lower.contains("at most one"));
        assert!(!lower.contains("do not emit"));
        assert!(!lower.contains("do not narrate"));
        assert!(!p.contains("TDA is internal eyes"));
        let packed = format!("<bos>{}<|turn>user\nSay hi.<turn|>\n", p);
        assert!(packed_prompt_has_emit_channel(&packed));
        assert!(!packed_prompt_has_legacy_panel(&packed));
        assert!(!p.contains("<|turn>user\nTags (emit only)"));
        assert!(
            !p.contains("<tools>"),
            "do not teach file tools this binary cannot run"
        );
    }
}

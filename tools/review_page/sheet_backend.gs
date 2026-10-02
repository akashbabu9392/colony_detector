/**
 * Shared store for the colony review page: a Google Sheet.
 *
 * Setup (once): create a Google Sheet, Extensions -> Apps Script, replace the
 * code with this file, Deploy -> New deployment -> Web app,
 * Execute as: Me, Who has access: Anyone -> Deploy, allow access, and copy the
 * Web app URL (ends in /exec). Build the page with
 *   python tools/answer_key.py kit ... --sheet-url <that URL>
 *
 * Every decision is appended to the "decisions" tab (time, reviewer, spot,
 * decision), so nothing is ever overwritten and the full history stays. The
 * page reads the latest decision per spot.
 */

const TAB = "decisions";
const ALLOWED = ["colony", "not", "small", "unsure", "countable", "overgrown", "tntc"];

function sheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(TAB);
  if (!sh) {
    sh = ss.insertSheet(TAB);
    sh.appendRow(["time", "reviewer", "spot", "decision"]);
    sh.setFrozenRows(1);
  }
  return sh;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  const body = JSON.parse((e && e.postData && e.postData.contents) || "{}");
  const who = String(body.reviewer || "").slice(0, 60);
  const now = new Date().toISOString();
  const rows = [];
  Object.keys(body.calls || {}).forEach(function (k) { rows.push([now, who, String(k), String(body.calls[k])]); });
  Object.keys(body.flags || {}).forEach(function (k) { rows.push([now, who, "plate:" + k, String(body.flags[k])]); });
  const clean = rows.filter(function (r) { return ALLOWED.indexOf(r[3]) >= 0 && /^(plate:)?\d+$/.test(r[2]); });
  if (clean.length) {
    const lock = LockService.getScriptLock();
    lock.waitLock(20000);
    try {
      const sh = sheet_();
      sh.getRange(sh.getLastRow() + 1, 1, clean.length, 4).setNumberFormat("@").setValues(clean);
    } finally {
      lock.releaseLock();
    }
  }
  return json_({ ok: true, saved: clean.length });
}

function doGet() {
  const sh = sheet_();
  const n = sh.getLastRow() - 1;
  const latest = {};
  if (n > 0) {
    sh.getRange(2, 1, n, 4).getDisplayValues().forEach(function (r) {
      latest[String(r[2])] = [String(r[3]), String(r[1]), String(r[0])];
    });
  }
  return json_({ latest: latest, rows: n });
}

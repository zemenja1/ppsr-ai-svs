#!/usr/bin/env bash
# Usage: make_bjt_out_job.sh <ClassName> <samples.txt> <base.mph> <out.java>
# Bipolar transistor with self-heating, common-emitter output characteristics I_C(V_C) at fixed base current.
# Same base model and samples as make_bjt_job.sh. Per sample:
#   std5  initialization as in make_bjt_job.sh (continuation in p to the sample's T0, doping and V_C)
#   std6  isothermal Gummel sweep at the sample's V_C (only used as the starting point, not emitted)
#   then for each base current I_B the base contact is switched to current drive (feature mc4 of the
#   library model, used by its studies 3 and 4) and V_C is swept from the sample's V_C down to 0 V and up
#   to 5 V, starting from the std6 point with the closest base current:
#   sto1  semiconductor at constant T0 (tag OISO)
#   sto2  semiconductor and heat transfer fully coupled, starting from the first sto1 point (tag OHOT)
# sto1/sto2 are new studies with default solvers: the solver sequences stored with studies 3 and 4 belong to
# the isothermal library model and do not converge here.
# samples.txt lines: id;T0_K;R_K_W;VC_V;NB_cm3;Nepi_cm3
cls="$1"; dev="$2"; base="$3"; out="$4"
{
cat <<EOF
import com.comsol.model.*;
import com.comsol.model.util.*;
public class $cls {
  static final String BASE = "$base";
  static final String[][] DEV = {
EOF
while IFS=';' read -r id t0 r vc nb nepi; do
  [ -z "$id" ] && continue
  echo "    {\"$id\",\"$t0\",\"$r\",\"$vc\",\"$nb\",\"$nepi\"},"
done < "$dev"
cat <<'EOF'
  };
  static final String P_LIST = "0 0.25 0.5 0.75 1 1.25 1.5 1.75 2";
  static final double[] IB_LIST = {1e-5, 1e-4, 1e-3};
  static final double[] VC_GRID = {0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5};
  static void out(String s) { System.out.println(s); System.out.flush(); }
  static String oneLine(Exception e) {
    StringBuilder b = new StringBuilder();
    for (char c : String.valueOf(e.getMessage()).toCharArray()) b.append(c < 32 ? (char) 32 : (c == (char) 44 ? (char) 59 : c));
    String s = b.toString();
    return s.length() > 200 ? s.substring(0, 200) : s;
  }
  static String dsOf(Model m, String study) {
    for (String s : m.sol().tags()) if (m.sol(s).study().equals(study))
      for (String ds : m.result().dataset().tags()) { try { if (m.result().dataset(ds).getString("solution").equals(s)) return ds; } catch (Exception e) {} }
    return null;
  }
  // emits one line per converged V_C step: tag,id,T0,R,VC,NB,Nepi,I_B,V_C,I_E,I_C,I_B(computed),V_BE,Tmax
  static int emit(Model m, String st, String tag, String[] d, double ib) {
    String ds = dsOf(m, st);
    NumericalFeature ev = m.result().numerical("ev"), mx = m.result().numerical("mx");
    ev.set("data", ds); ev.set("expr", new String[]{"V_C", "semi.I0_1", "semi.I0_3", "semi.I0_4", "semi.V0_4"});
    mx.set("data", ds);
    double[][] r = ev.getReal(), t = mx.getReal();
    for (int i = 0; i < r[0].length; i++)
      out(String.format(java.util.Locale.ROOT, "%s,%s,%s,%s,%s,%s,%s,%.3g,%.6g,%.9e,%.9e,%.9e,%.6f,%.6f", tag, d[0], d[1], d[2], d[3], d[4], d[5],
          ib, r[0][i], r[1][i], r[2][i], r[3][i], r[4][i], t[0][i]));
    return r[0].length;
  }
  static void newStudy(Model m, String st, boolean ht) {
    m.study().create(st); m.study(st).create("stat", "Stationary");
    StudyFeature s = m.study(st).feature("stat");
    s.set("activate", new String[]{"semi", "on", "ht", ht ? "on" : "off", "frame:spatial1", "on", "frame:material1", "on"});
    s.set("useparam", "on"); s.set("pname", new String[]{"I_B", "V_C"}); s.set("punit", new String[]{"A", "V"});
    s.set("plistarr", new String[]{"1e-5", "0"}); s.set("sweeptype", "filled"); s.set("pcontinuation", "V_C");
    s.set("useinitsol", "on"); s.set("initmethod", "sol");
  }
  // V_C list of one sweep: the start value, then the grid points beyond it in one direction
  static String list(double vc, boolean up) {
    StringBuilder b = new StringBuilder(String.valueOf(vc));
    if (up) { for (double v : VC_GRID) if (v > vc + 1e-9) b.append(" ").append(v); }
    else { for (int i = VC_GRID.length - 1; i >= 0; i--) if (VC_GRID[i] < vc - 1e-9) b.append(" ").append(VC_GRID[i]); }
    return b.toString();
  }
  // runs one sweep and emits its converged steps; returns -1 if the sweep did not finish
  static int sweep(Model m, String st, String tag, String[] d, double ib, String vcl, String init, String solnum) {
    StudyFeature s = m.study(st).feature("stat");
    s.set("plistarr", new String[]{String.valueOf(ib), vcl}); s.set("initstudy", init); s.set("solnum", solnum);
    boolean ok = true;
    try { m.study(st).run(); } catch (Exception e) { ok = false; out(tag + "FAIL," + d[0] + "," + ib + "," + vcl + "," + oneLine(e)); }
    int n = 0;
    try { n = emit(m, st, tag, d, ib); } catch (Exception e) { out(tag + "NODATA," + d[0] + "," + ib); }
    return ok ? n : -1;
  }
  public static Model run() {
    Model m;
    try { m = ModelUtil.load("m", BASE); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    m.result().numerical().create("ev", "EvalGlobal");
    NumericalFeature mx = m.result().numerical().create("mx", "MaxSurface");
    mx.selection().all(); mx.set("expr", new String[]{"T"});
    newStudy(m, "sto1", false); newStudy(m, "sto2", true);
    NumericalFeature g6 = m.result().numerical().create("g6", "EvalGlobal");
    g6.set("data", "dset6"); g6.set("expr", new String[]{"semi.I0_2"});
    for (String[] d : DEV) {
      long t0 = System.currentTimeMillis();
      out("START," + d[0]);
      int ni = 0, nh = 0;
      try {
        double vc = Double.parseDouble(d[3]);
        m.physics("semi").feature("mc4").active(false);
        m.param().set("p", "0");
        m.param().set("hs", "min(p,1)");
        m.param().set("R", d[2] + "[K/W]");
        m.param().set("V_C", "(0.5+max(p-1,0)*(" + (vc - 0.5) + "))[V]");
        m.param().set("T0", "(293.15+hs*(" + (Double.parseDouble(d[1]) - 293.15) + "))[K]");
        m.param().set("N_B", "9e17[1/cm^3]*(" + (Double.parseDouble(d[4]) / 9e17) + ")^hs");
        m.param().set("N_epi", "2e16[1/cm^3]*(" + (Double.parseDouble(d[5]) / 2e16) + ")^hs");
        StudyFeature s5 = m.study("std5").feature("stat");
        s5.set("pname", new String[]{"V_B", "V_E", "p"}); s5.set("punit", new String[]{"V", "V", ""});
        s5.set("plistarr", new String[]{"0", "0", P_LIST});
        s5.set("sweeptype", "filled"); s5.set("pcontinuation", "p");
        m.study("std5").run();
        m.param().set("p", "2");
        m.study("std6").run();
        double[] ibg = g6.getReal()[0];
        m.param().set("V_C", vc + "[V]");   // V_C is now a sweep parameter
        m.physics("semi").feature("mc4").active(true);
        for (double ib : IB_LIST) {
          int k = 0; double best = 1e9;
          for (int i = 0; i < ibg.length; i++) { double dd = Math.abs(Math.log(Math.abs(ibg[i]) + 1e-30) - Math.log(ib)); if (dd < best) { best = dd; k = i; } }
          for (boolean up : new boolean[]{false, true}) {
            String vcl = list(vc, up);
            if (vcl.indexOf(' ') < 0) continue;
            int n = sweep(m, "sto1", "OISO", d, ib, vcl, "std6", String.valueOf(k + 1));
            ni += Math.abs(n);
            if (n > 0) nh += Math.abs(sweep(m, "sto2", "OHOT", d, ib, vcl, "sto1", "1"));
          }
        }
        m.physics("semi").feature("mc4").active(false);
        out(String.format(java.util.Locale.ROOT, "DONE,%s,%d,%d,%.1f", d[0], ni, nh, (System.currentTimeMillis() - t0) / 1000.0));
      } catch (Exception e) {
        m.physics("semi").feature("mc4").active(false);
        out(String.format(java.util.Locale.ROOT, "FAIL,%s,%.1f,%s", d[0], (System.currentTimeMillis() - t0) / 1000.0, oneLine(e)));
      }
    }
    out("WORKER_FINISHED");
    return m;
  }
  public static void main(String[] a) { run(); }
}
EOF
} > "$out"

#!/usr/bin/env bash
# Usage: make_bjt_job.sh <ClassName> <samples.txt> <base.mph> <out.java>
# Bipolar transistor with self-heating: Application Library model bipolar_transistor_thermal, used unchanged
# except for the parameters below. The base model must contain the stored solution of study 1 (std1 does not
# converge from the default initial values in batch mode; the thermal studies start from its solution).
# Per sample, the studies of the library model are run in their original chain:
#   std5  incomplete ionization, V_C ramp 0.5 V ... V_C at V_B = 0 (initialization, not emitted)
#   std6  Gummel sweep V_B = 0.4 ... 1.1 V, semiconductor at constant T0 (tag ISO)
#   std7  the same sweep, fully coupled with heat transfer (tag HOT, with the maximum temperature)
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
  static void out(String s) { System.out.println(s); System.out.flush(); }
  static String oneLine(Exception e) {
    StringBuilder b = new StringBuilder();
    for (char c : String.valueOf(e.getMessage()).toCharArray()) b.append(c < 32 ? (char) 32 : (c == (char) 44 ? (char) 59 : c));
    String s = b.toString();
    return s.length() > 200 ? s.substring(0, 200) : s;
  }
  // emits one line per converged V_B step: tag,id,T0,R,VC,NB,Nepi,V_B,I_E,I_B,I_C,Tmax
  static int emit(Model m, String ds, String tag, String[] d) {
    NumericalFeature ev = m.result().numerical("ev"), mx = m.result().numerical("mx");
    ev.set("data", ds); mx.set("data", ds);
    double[][] r = ev.getReal(), t = mx.getReal();
    for (int i = 0; i < r[0].length; i++)
      out(String.format(java.util.Locale.ROOT, "%s,%s,%s,%s,%s,%s,%s,%.6g,%.9e,%.9e,%.9e,%.6f", tag, d[0], d[1], d[2], d[3], d[4], d[5],
          r[0][i], r[1][i], r[2][i], r[3][i], t[0][i]));
    return r[0].length;
  }
  public static Model run() {
    Model m;
    try { m = ModelUtil.load("m", BASE); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    NumericalFeature ev = m.result().numerical().create("ev", "EvalGlobal");
    ev.set("expr", new String[]{"V_B", "semi.I0_1", "semi.I0_2", "semi.I0_3"});
    NumericalFeature mx = m.result().numerical().create("mx", "MaxSurface");
    mx.selection().all(); mx.set("expr", new String[]{"T"});
    for (String[] d : DEV) {
      long t0 = System.currentTimeMillis();
      out("START," + d[0]);
      int n6 = 0, n7 = 0;
      try {
        // The initialization study starts from the stored solution of the library device (T0 = 293.15 K,
        // N_B = 9e17, N_epi = 2e16 cm^-3, V_C = 0.5 V, V_B = 0). A direct jump to other values diverges, so std5
        // continues in one parameter p = 0 ... 2 (automatic step control): p = 0 ... 1 moves T0 and the doping
        // from the library values to this sample, p = 1 ... 2 ramps the collector voltage to this sample's V_C.
        m.param().set("p", "0");
        m.param().set("hs", "min(p,1)");
        m.param().set("R", d[2] + "[K/W]");
        m.param().set("V_C", "(0.5+max(p-1,0)*(" + (Double.parseDouble(d[3]) - 0.5) + "))[V]");
        m.param().set("T0", "(293.15+hs*(" + (Double.parseDouble(d[1]) - 293.15) + "))[K]");
        m.param().set("N_B", "9e17[1/cm^3]*(" + (Double.parseDouble(d[4]) / 9e17) + ")^hs");
        m.param().set("N_epi", "2e16[1/cm^3]*(" + (Double.parseDouble(d[5]) / 2e16) + ")^hs");
        StudyFeature s5 = m.study("std5").feature("stat");
        s5.set("pname", new String[]{"V_B", "V_E", "p"}); s5.set("punit", new String[]{"V", "V", ""});
        s5.set("plistarr", new String[]{"0", "0", P_LIST});
        s5.set("sweeptype", "filled"); s5.set("pcontinuation", "p");
        m.study("std5").run();
        m.param().set("p", "2");   // studies 6 and 7 at this sample's values
        try { m.study("std6").run(); } catch (Exception e) { out("ISOFAIL," + d[0] + "," + oneLine(e)); }
        try { n6 = emit(m, "dset6", "ISO", d); } catch (Exception e) { out("ISONODATA," + d[0]); }
        try { m.study("std7").run(); } catch (Exception e) {
          // retry from the first isothermal point (same V_B = 0.4 V) instead of the V_B = 0 solution of std5
          out("HOTRETRY," + d[0] + "," + oneLine(e));
          StudyFeature s7 = m.study("std7").feature("stat");
          s7.set("initstudy", "std6"); s7.set("solnum", "1");
          try { m.study("std7").run(); } catch (Exception e2) { out("HOTFAIL," + d[0] + "," + oneLine(e2)); }
          s7.set("initstudy", "std5"); s7.set("solnum", "auto");
        }
        try { n7 = emit(m, "dset7", "HOT", d); } catch (Exception e) { out("HOTNODATA," + d[0]); }
        out(String.format(java.util.Locale.ROOT, "DONE,%s,%d,%d,%.1f", d[0], n6, n7, (System.currentTimeMillis() - t0) / 1000.0));
      } catch (Exception e) {
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

"""Generate a COMSOL batch class running E-core transformer cases (Study 2 = case 1 setup, Np = Ns = 300, Rp = 100 ohm)
with varied source amplitude Vac (V), frequency nu (Hz) and load resistance Rs (ohm).
Usage: python gen_transformer.py <Class> <cases.json>, then  comsolbatch -inputfile <Class>.class -np 4 -nosave > log
TR_MPH: the Application Library model ACDC_Module/Devices,_Inductive/ecore_transformer.mph after computing all its
studies in the COMSOL GUI and saving it (the library file has no stored solutions; the batch runs need the coil
geometry analysis of study 1). COMSOLCOMPILE: path of comsolcompile."""
import os, sys, json, subprocess
MPH = os.environ.get("TR_MPH", "ecore_transformer_solved.mph").replace(chr(92), "/")
TEMPLATE = r'''
import com.comsol.model.*;
import com.comsol.model.util.*;
public class %(cls)s {
  static void out(String s) { System.out.println(s); System.out.flush(); }
  static void dump(Model m, String num, String tab, String key, String cid) {
    m.result().table(tab).clearTableData();
    m.result().numerical(num).setResult();
    double[][] d = m.result().table(tab).getReal();
    for (double[] r : d) { StringBuilder sb = new StringBuilder(key + "," + cid); for (double v : r) sb.append(",").append((float) v); out(sb.toString()); }
  }
  public static void main(String[] a) {
    Model m;
    try { m = ModelUtil.load("tr", "%(mph)s"); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    var c = m.component("comp1");
    // results: circuit quantities (time series), B in the core, core loss (cycle average)
    m.result().numerical().create("gv9", "EvalGlobal");
    m.result().numerical("gv9").set("data", "dset2");
    m.result().numerical("gv9").set("expr", new String[]{"mf.VCoil_1", "mf.ICoil_1", "mf.VCoil_2", "mf.ICoil_2"});
    m.result().table().create("tgv", "Table"); m.result().numerical("gv9").set("table", "tgv");
    m.result().numerical().create("mx9", "MaxVolume");
    m.result().numerical("mx9").set("data", "dset2"); m.result().numerical("mx9").selection().set(new int[]{2});
    m.result().numerical("mx9").set("expr", new String[]{"mf.normB"});
    m.result().table().create("tmx", "Table"); m.result().numerical("mx9").set("table", "tmx");
    m.result().numerical().create("av9", "AvVolume");
    m.result().numerical("av9").set("data", "dset2"); m.result().numerical("av9").selection().set(new int[]{2});
    m.result().numerical("av9").set("expr", new String[]{"mf.normB", "mf.normH"});
    m.result().table().create("tav", "Table"); m.result().numerical("av9").set("table", "tav");
    m.result().numerical().create("lq9", "IntVolume");
    m.result().numerical("lq9").set("data", "dset4"); m.result().numerical("lq9").selection().set(new int[]{2});
    m.result().numerical("lq9").set("expr", new String[]{"mf.Qh"});
    m.result().table().create("tlq", "Table"); m.result().numerical("lq9").set("table", "tlq");
    m.result().numerical().create("lr9", "IntVolume");
    m.result().numerical("lr9").set("data", "dset4"); m.result().numerical("lr9").selection().set(new int[]{3, 4, 5, 6, 7, 8, 9, 10});
    m.result().numerical("lr9").set("expr", new String[]{"mf.Qh"});
    m.result().table().create("tlr", "Table"); m.result().numerical("lr9").set("table", "tlr");
    m.result().numerical().create("rc9", "EvalGlobal");
    m.result().numerical("rc9").set("data", "dset2");
    m.result().numerical("rc9").set("expr", new String[]{"mf.RCoil_1", "mf.RCoil_2"});
    m.result().numerical("rc9").setIndex("looplevelinput", "last", 0);
    m.result().table().create("trc", "Table"); m.result().numerical("rc9").set("table", "trc");
    String[][] cases = %(cases)s;
    for (String[] k : cases) {
      out("CASE," + String.join(";", k));
      try {
        c.variable("var1").set("Vac", k[1] + "[V]"); c.variable("var1").set("Rs", k[3] + "[ohm]");
        m.param().set("nu", k[2] + "[Hz]");
        double T = 1.0 / Double.parseDouble(k[2]);
        String tl = "range(0," + (T / 40) + "," + (2.5 * T) + ")";
        m.study("std2").feature("time").set("tlist", tl);
        try { m.sol("sol2").feature("t1").set("tlist", tl); } catch (Exception e) { out("WARN,tlist " + e); }
        m.study("std4").feature("emloss").set("emT0", "" + T);
        m.study("std4").feature("emloss").set("fftendtime", "" + (2.5 * T));
        m.study("std4").feature("emloss").set("lossendtime", "" + T);
        long t0 = System.currentTimeMillis();
        m.study("std2").run();
        long t1 = System.currentTimeMillis();
        m.study("std4").run();
        out("SOLVETIME," + k[0] + "," + (t1 - t0) / 1000.0 + "," + (System.currentTimeMillis() - t1) / 1000.0);
        dump(m, "gv9", "tgv", "GV", k[0]); dump(m, "mx9", "tmx", "MX", k[0]); dump(m, "av9", "tav", "AV", k[0]);
        dump(m, "lq9", "tlq", "LQ", k[0]); dump(m, "lr9", "tlr", "LR", k[0]); dump(m, "rc9", "trc", "RC", k[0]);
      } catch (Exception e) { out("FAIL," + k[0] + "," + e.toString().replace("\n", " ")); }
    }
  }
}
'''
if __name__ == "__main__":
    cls, spec = sys.argv[1], json.load(open(sys.argv[2]))
    cases = [[c["cid"], f'{c["Vac"]:.6g}', f'{c["nu"]:.6g}', f'{c["Rs"]:.6g}'] for c in spec]
    java = TEMPLATE % dict(mph=MPH, cls=cls, cases="{" + ",".join("{" + ",".join(json.dumps(v) for v in c) + "}" for c in cases) + "}")
    open(cls + ".java", "w").write(java)
    B = os.environ.get("COMSOLCOMPILE", "C:/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe")
    r = subprocess.run([B, cls + ".java"], capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-800:])

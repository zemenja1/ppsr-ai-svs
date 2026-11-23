"""Generate a COMSOL batch Java class running a list of iron-ribbon cases (constant field H0 from t = 0, or from
t = trel after a relaxation at H = 0; wall = 2: wall nucleated at the left end, as in step 6/7 of the notebook).
Usage: python gen_ribbon.py <ClassName> <cases.json>  ->  <ClassName>.java/.class; run it with comsolbatch -inputfile.
The model (COMSOL 6.3 + the Micromagnetics Module plugin of Fudan University, not included) is set by RIBBON_MPH."""
import os, sys, json, math, subprocess
MPH = os.environ.get("RIBBON_MPH", "ribbon.mph").replace(chr(92), "/")
TEMPLATE = r'''
import com.comsol.model.*;
import com.comsol.model.util.*;
public class %(cls)s {
  static void out(String s) { System.out.println(s); System.out.flush(); }
  public static void main(String[] a) {
    Model m;
    try { m = ModelUtil.load("rib", "%(mph)s"); } catch (java.io.IOException e) { throw new RuntimeException(e); }
    m.label("iron_ribbon_step");
    m.param().set("Hext", "0[A/m]");
    m.param().set("H0", "2e4[A/m]");
    m.param().set("f0", "10[MHz]");
    m.param().set("Kfe", "4.37e4");
    m.param().set("epin", "0.4");
    m.param().set("lam", "64[nm]");
    m.param().set("kend", "3");
    m.param().set("alpha0", "1");
    m.param().set("d0", "21[nm]");
    m.param().set("Afe", "1.96e-11");
    m.param().set("x0", "150[nm]");
    m.param().set("wall", "1");
    m.param().set("th0", "1[deg]");
    m.param().set("trel", "0[ns]");
    String Hx = "if(trel>0[s],H0*flc1hs(t-trel,0.1[ns]),H0)";
    var fe = m.component("comp1").physics("mm").feature("LLG1");
    fe.set("K", "Kfe*(1-epin*cos(2*pi*(x-150[nm])/lam))");
    fe.set("alpha", "alpha0"); fe.set("A", "Afe");
    fe.set("H", new String[]{Hx, "0", "0"});
    fe.label("LLG Fe with pinning");
    var he = m.component("comp1").physics("mm").feature("LLG2");
    he.selection().set(new int[]{1, 5});
    he.set("Ms", "1.71e6"); he.set("A", "Afe"); he.set("K", "kend*Kfe"); he.set("alpha", "alpha0");
    he.set("H", new String[]{Hx, "0", "0"});
    he.label("LLG harder Fe (end stops)");
    m.component("comp1").physics("mm").feature("init1").set("m", new String[]{"if(wall>1.5,if(x<30[nm],1,if(x<40[nm],0,-1)),if(wall>0.5,-tanh((x-x0)/d0),-cos(th0)))", "if(wall>1.5,if(x>=30[nm]&&x<40[nm],1,0),if(wall>0.5,1/cosh((x-x0)/d0),sin(th0)))", "0"});
    m.component("comp1").physics("mm").feature("init2").active(false);
    m.component("comp1").physics("mm").feature("init3").active(false);
    m.result().numerical().create("av9", "AvVolume");
    m.result().numerical("av9").set("data", "dset1");
    m.result().numerical("av9").selection().all();
    m.result().numerical("av9").set("expr", new String[]{"mX", "mY", "mZ", "mX*(dom>1)*(dom<5)", "sqrt(mX^2+mY^2+mZ^2)"});
    m.result().table().create("tav", "Table");
    m.result().numerical("av9").set("table", "tav");
    m.result().numerical().create("mx9", "MaxVolume");
    m.result().numerical("mx9").set("data", "dset1");
    m.result().numerical("mx9").selection().all();
    m.result().numerical("mx9").set("expr", new String[]{"abs(sqrt(mX^2+mY^2+mZ^2)-1)"});
    m.result().table().create("tmx", "Table");
    m.result().numerical("mx9").set("table", "tmx");
    m.result().dataset().create("cpt9", "CutPoint3D");
    m.result().dataset("cpt9").set("data", "dset1");
    m.result().dataset("cpt9").set("pointx", "range(0,2,300)");
    m.result().dataset("cpt9").set("pointy", "2.5");
    m.result().dataset("cpt9").set("pointz", "0.5");
    m.result().numerical().create("pev9", "EvalPoint");
    m.result().numerical("pev9").set("data", "cpt9");
    m.result().numerical("pev9").set("expr", new String[]{"mX", "mY", "mZ"});
    m.result().table().create("tpr", "Table");
    m.result().numerical("pev9").set("table", "tpr");
    String[][] cases = %(cases)s;
    for (String[] c : cases) {
      out("CASE," + String.join(",", c));
      try {
        m.param().set("H0", c[1] + "[A/m]"); m.param().set("f0", c[2] + "[MHz]"); m.param().set("Kfe", c[3]);
        m.param().set("epin", c[4]); m.param().set("lam", c[5] + "[nm]"); m.param().set("kend", c[6]);
        m.param().set("alpha0", c[7]); m.param().set("d0", c[8] + "[nm]");
        m.param().set("Afe", c[11]); m.param().set("x0", c[12] + "[nm]"); m.param().set("wall", c[13]); m.param().set("trel", c[14] + "[ns]");
        m.study("std1").feature("time").set("tlist", c[9]);
        try { m.sol("sol1").feature("t1").set("tlist", c[9]); } catch (Exception e) { out("WARN,tlist " + e); }
        try { m.sol("sol1").feature("t1").set("rtol", c[10]); m.sol("sol1").feature("t1").set("tolfact", "1"); } catch (Exception e) { out("WARN,rtol " + e); }
        long t0 = System.currentTimeMillis();
        m.study("std1").run();
        out("SOLVETIME," + c[0] + "," + (System.currentTimeMillis() - t0) / 1000.0);
        for (String[] tn : new String[][]{{"av9", "tav", "AV"}, {"mx9", "tmx", "MX"}, {"pev9", "tpr", "PR"}}) {
          m.result().table(tn[1]).clearTableData();
          m.result().numerical(tn[0]).setResult();
          double[][] d = m.result().table(tn[1]).getReal();
          for (double[] r : d) { StringBuilder sb = new StringBuilder(tn[2] + "," + c[0]); for (double v : r) sb.append(",").append((float) v); out(sb.toString()); }
        }
      } catch (Exception e) { out("FAIL," + c[0] + "," + e.toString().replace("\n", " ")); }
    }
  }
}
'''
def case(cid, H0, K0=4.37e4, A=1.96e-11, eps=0.4, lam=64, kend=20, alpha=1.0, tend=20, dt=0.1, d0=None, x0=86, wall=1, rtol=0.001, trel=0):
    """Constant field H0 (A/m) from t = 0. K0 = H_K (A/m), A in A*m (plugin field units)."""
    if d0 is None: d0 = math.sqrt(A / K0) * 1e9
    tlist = f"range(0,{dt:g},{tend + trel:g})"
    return [cid, f"{H0:g}", "1", f"{K0:g}", f"{eps:g}", f"{lam:g}", f"{kend:g}", f"{alpha:g}", f"{d0:.4g}", tlist, f"{rtol:g}", f"{A:g}", f"{x0:g}", f"{wall:g}", f"{trel:g}"]

if __name__ == "__main__":
    cls, spec = sys.argv[1], json.load(open(sys.argv[2]))
    cases = [case(**c) for c in spec]
    java = TEMPLATE % dict(mph=MPH, cls=cls, cases="{" + ",".join("{" + ",".join(json.dumps(v) for v in c) + "}" for c in cases) + "}")
    open(cls + ".java", "w").write(java)
    B = os.environ.get("COMSOLCOMPILE", "C:/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe")
    r = subprocess.run([B, cls + ".java"], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-500:])

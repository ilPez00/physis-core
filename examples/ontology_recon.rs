use physis_core::ontology::OntologyLoader;
use std::collections::HashMap;

fn main() {
    let o = OntologyLoader::load_all();
    let mut per_domain: HashMap<String, usize> = HashMap::new();
    let mut per_cell: HashMap<(String, String), usize> = HashMap::new();
    let mut total = 0;
    let mut no_domain_mode = 0;
    for def in o.classification_domains() {
        total += 1;
        match (&def.domain, &def.mode) {
            (Some(d), Some(m)) => {
                *per_domain.entry(d.clone()).or_default() += 1;
                *per_cell.entry((d.clone(), m.clone())).or_default() += 1;
            }
            _ => no_domain_mode += 1,
        }
    }
    println!("total entries: {total}  (no domain/mode: {no_domain_mode})");
    println!("domains: {}", per_domain.len());
    let mut pd: Vec<_> = per_domain.into_iter().collect();
    pd.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (d, c) in &pd {
        println!("  {d:<12} {c}");
    }
    println!("cells (domain,mode): {}", per_cell.len());
    let mut pc: Vec<_> = per_cell.into_iter().collect();
    pc.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    println!("top 10 cells by count:");
    for ((d, m), c) in pc.iter().take(10) {
        println!("  {d}/{m}: {c}");
    }
    println!("cells with >=5 entries: {}", pc.iter().filter(|(_, c)| *c >= 5).count());
    println!("cells with 1 entry: {}", pc.iter().filter(|(_, c)| *c == 1).count());
}

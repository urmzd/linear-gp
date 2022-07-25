pub fn argmax<I: Iterator<Item = f64>>(iter: I) -> Option<usize> {
    let mut current_max = None;
    let mut max_index = -1;

    let index = 0;

    for item in iter {
        if Some(item) > current_max {
            current_max = Some(item);
            max_index = index;
        }
    }

    if max_index < 0 {
        None
    } else {
        Some(max_index as usize)
    }
}

pub fn max_val<I: Iterator<Item = f64>>(iter: I) -> Option<f64> {
    iter.reduce(f64::max)
}

use test_log::test;

pub fn argmax<I: Iterator<Item = f64>>(iter: I) -> Option<usize> {
    let mut current_max = None;
    let mut max_index = -1;

    let mut index = 0;

    for item in iter {
        if Some(item) > current_max {
            current_max = Some(item);
            max_index = index;
        }
        index += 1;
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

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn given_iterator_of_floats_when_argmax_then_max_index_is_returned() {
        let values = [0., 1., 2.];
        let argmax = argmax(values.iter().copied());

        pretty_assertions::assert_eq!(argmax, Some(2));
    }
}

pub fn argmax<I: Iterator<Item = f64>>(iter: I) -> Option<usize> {
    let mut current_max = None;
    let mut max_index = None;

    for (index, item) in iter.enumerate() {
        if Some(item) > current_max {
            current_max = Some(item);
            max_index = Some(index);
        }
    }

    max_index
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn given_iterator_of_floats_when_argmax_then_max_index_is_returned() {
        let values = [0., 1., 2.];
        let argmax = argmax(values.iter().copied());

        assert_eq!(argmax, Some(2));
    }
}

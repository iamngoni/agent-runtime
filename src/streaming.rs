pub fn should_flush_delta(buffer: &str) -> bool {
    let trimmed = buffer.trim_end();
    if trimmed.is_empty() {
        return false;
    }

    buffer.chars().count() >= 120
        || (buffer.chars().count() >= 40 && trimmed.ends_with('.'))
        || (buffer.chars().count() >= 40 && trimmed.ends_with('!'))
        || (buffer.chars().count() >= 40 && trimmed.ends_with('?'))
        || trimmed.ends_with("\n\n")
}

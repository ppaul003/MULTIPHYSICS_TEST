#include "TextEntry.h"

#include <limits>
#include <charconv>
#include <cmath>

namespace {
	constexpr unsigned char kEnter = 13;
	constexpr unsigned char kEscape = 27;
	constexpr unsigned char kBackspace = 8;
	constexpr unsigned char kDeleteBackspace = 127;
}

std::string TextEntrySession::formatReal(double value) {
    char text[768];
    const auto result = std::to_chars(text, text + sizeof(text), value, std::chars_format::fixed);
    return result.ec == std::errc{} ? std::string(text, result.ptr) : std::string{};
}

bool TextEntrySession::beginNonnegativeReal(const std::string& prompt, double initialValue) {
    reset();
    if (!std::isfinite(initialValue) || initialValue < 0) {
        m_statusMessage = "A finite nonnegative default is required.";
        return false;
    }
    m_mode = TextEntryMode::NonnegativeReal;
    m_active = m_replaceNumeric = true;
    m_prompt = prompt;
    m_buffer = formatReal(initialValue);
    m_maximumLength = 768;
    return true;
}

bool TextEntrySession::beginSignedInteger(const std::string& prompt, int minimum, int maximum, int initialValue) {
    reset();
    if (minimum > maximum || initialValue < minimum || initialValue > maximum) {
        m_statusMessage = "Signed-integer range/default is invalid.";
        return false;
    }
    m_mode = TextEntryMode::SignedInteger;
    m_active = m_replaceNumeric = true;
    m_prompt = prompt;
    m_buffer = std::to_string(initialValue);
    m_signedMinimum = minimum;
    m_signedMaximum = maximum;
    m_maximumLength = 11;
    return true;
}

TextEntryAction TextEntrySession::handleNumericKey(unsigned char rawKey) {
    std::string candidate = m_replaceNumeric ? std::string{} : m_buffer;
    const bool digit = rawKey >= '0' && rawKey <= '9';
    const bool decimal = m_mode == TextEntryMode::NonnegativeReal && rawKey == '.' && candidate.find('.') == std::string::npos;
    const bool sign = m_mode == TextEntryMode::SignedInteger && rawKey == '-' && candidate.empty();
    if (!digit && !decimal && !sign) return reject("Character is not valid for this numeric field.");
    if (candidate.size() >= m_maximumLength) return reject("Numeric entry exceeds maximum length.");
    candidate.push_back(static_cast<char>(rawKey));
    m_buffer = candidate;
    m_replaceNumeric = false;
    m_statusMessage = "Entry changed.";
    return TextEntryAction::Changed;
}

bool TextEntrySession::tryGetCommittedReal(double& value) const {
    if (!m_hasCommittedReal) return false;
    value = m_committedReal; return true;
}
bool TextEntrySession::tryGetCommittedSigned(int& value) const {
    if (!m_hasCommittedSigned) return false;
    value = m_committedSigned; return true;
}

bool TextEntrySession::beginUnsignedInteger(
	const std::string& prompt,
	unsigned int minimum,
	unsigned int maximum,
	unsigned int initialValue) {

	return beginUnsignedIntegerSession(
		prompt,
		minimum,
		maximum,
		true,
		initialValue
	);
}

bool TextEntrySession::beginUnsignedInteger(
	const std::string& prompt,
	unsigned int minimum,
	unsigned int maximum) {

	return beginUnsignedIntegerSession(
		prompt,
		minimum,
		maximum,
		false,
		0
	);
}

bool TextEntrySession::beginUnsignedIntegerSession(
	const std::string& prompt,
	unsigned int minimum,
	unsigned int maximum,
	bool hasDefault,
	unsigned int defaultValue) {

	reset();

	if (minimum > maximum) {
		m_statusMessage = "Unsigned-integer range is invalid.";
		return false;
	}

	if (hasDefault &&
		(defaultValue < minimum || defaultValue > maximum)) {

		m_statusMessage = "Unsigned-integer default is outside the allowed range.";
		return false;
	}

	m_active = true;
	m_mode = TextEntryMode::UnsignedInteger;
	m_prompt = prompt;
	m_maximumLength = decimalDigitCount(maximum);
	m_unsignedMinimum = minimum;
	m_unsignedMaximum = maximum;
	m_unsignedDefault = defaultValue;
	m_hasUnsignedDefault = hasDefault;
	
	if (hasDefault) {
		m_buffer = 
			std::to_string(defaultValue);
	}

	m_statusMessage = "Enter an unsigned integer.";

	return true;
}

bool TextEntrySession::beginAssetName(
	const std::string& prompt,
	const std::string& initialValue,
	std::size_t maximumLength) {

	return beginTextSession(
		TextEntryMode::AssetName,
		prompt,
		initialValue,
		maximumLength
	);
}

bool TextEntrySession::beginGeneralText(
	const std::string& prompt,
	const std::string& initialValue,
	std::size_t maximumLength) {

	return beginTextSession(
		TextEntryMode::GeneralText,
		prompt,
		initialValue,
		maximumLength
	);
}

bool TextEntrySession::beginTextSession(
	TextEntryMode mode,
	const std::string& prompt,
	const std::string& initialValue,
	std::size_t maximumLength) {

	reset();

	if (mode != TextEntryMode::AssetName &&
		mode != TextEntryMode::GeneralText) {

		m_statusMessage = "Unsupported text-entry mode.";
		return false;
	}

	if (maximumLength == 0) {
		m_statusMessage = "Maximum length must be greater than zero.";
		return false;
	}

	if (initialValue.size() > maximumLength) {
		m_statusMessage = "Initial value exceeds the maximum length.";
		return false;
	}

	if (!isValidInitialText(mode, initialValue)) {
		m_statusMessage = "Initial value contains unsupported characters.";
		return false;
	}

	m_active = true;
	m_mode = mode;
	m_prompt = prompt;
	m_buffer = initialValue;
	m_maximumLength = maximumLength;
	m_statusMessage =
		(mode == TextEntryMode::AssetName)
		? "Enter an asset name."
		: "Enter text.";

	return true;
}

TextEntryAction TextEntrySession::handleRawKey(unsigned char rawKey) {
	if (!m_active) {
		return TextEntryAction::None;
	}

	if (rawKey == kEscape) {
		cancel();
		return TextEntryAction::Cancelled;
	}

	if (rawKey == kEnter) {
		return commitActiveSession();
	}

	if (rawKey == kBackspace || rawKey == kDeleteBackspace) {
		if (m_buffer.empty()) {
			return reject("Nothing to delete.");
		}

		m_replaceNumeric = false;
		m_buffer.pop_back();
		m_statusMessage = "Entry changed.";
		return TextEntryAction::Changed;
	}

	switch (m_mode) {
    case TextEntryMode::NonnegativeReal:
    case TextEntryMode::SignedInteger:
        return handleNumericKey(rawKey);
	case TextEntryMode::UnsignedInteger:
		return handleUnsignedIntegerKey(rawKey);

	case TextEntryMode::AssetName:
		return handleAssetNameKey(rawKey);

	case TextEntryMode::GeneralText:
		return handleGeneralTextKey(rawKey);

	default:
	case TextEntryMode::None:
		return TextEntryAction::None;
	}
}

TextEntryAction
TextEntrySession::handleUnsignedIntegerKey(unsigned char rawKey) {
	if (rawKey < '0' || rawKey > '9') {
		return reject("Unsigned integers accept only digits 0-9.");
	}

	if (m_buffer.size() >= m_maximumLength) {
		return reject("Unsigned integer exceeds the maximum length.");
	}

	const std::string candidate =
		m_buffer + static_cast<char>(rawKey);

	unsigned int value = 0;
	if (!tryParseUnsigned(candidate, value)) {
		return reject("Unsigned integer overflow was prevented.");
	}

	if (value > m_unsignedMaximum) {
		return reject("Unsigned integer exceeds the configured maximum.");
	}

	m_buffer = candidate;
	m_statusMessage = "Entry changed.";
	return TextEntryAction::Changed;
}

TextEntryAction TextEntrySession::handleAssetNameKey(unsigned char rawKey) {
	if (isWindowsReservedFilenameCharacter(rawKey)) {
		return reject("Character is reserved by Windows filenames.");
	}

	if (!isAssetNameCharacter(rawKey)) {
		return reject("Asset names accept letters, digits, spaces, '_' and '-'.");
	}

	if (m_buffer.size() >= m_maximumLength) {
		return reject("Asset name exceeds the maximum length.");
	}

	m_buffer.push_back(static_cast<char>(rawKey));
	m_statusMessage = "Entry changed.";
	return TextEntryAction::Changed;
}

TextEntryAction TextEntrySession::handleGeneralTextKey(unsigned char rawKey) {
	if (!isPrintableAscii(rawKey)) {
		return reject("General text accepts printable ASCII characters.");
	}

	if (m_buffer.size() >= m_maximumLength) {
		return reject("Text exceeds the maximum length.");
	}

	m_buffer.push_back(static_cast<char>(rawKey));
	m_statusMessage = "Entry changed.";
	return TextEntryAction::Changed;
}

TextEntryAction TextEntrySession::commitActiveSession() {
	switch (m_mode) {
    case TextEntryMode::NonnegativeReal: {
        double value = 0.0;
        const auto result = std::from_chars(m_buffer.data(), m_buffer.data()+m_buffer.size(), value, std::chars_format::fixed);
        if (result.ec != std::errc{} || result.ptr != m_buffer.data()+m_buffer.size() || !std::isfinite(value) || value < 0)
            return reject("A finite nonnegative decimal is required.");
        m_committedReal = value;
        m_hasCommittedReal = true;
        m_committedText = m_normalizedText = formatReal(value);
        break;
    }
    case TextEntryMode::SignedInteger: {
        int value = 0;
        const auto result = std::from_chars(m_buffer.data(), m_buffer.data()+m_buffer.size(), value);
        if (result.ec != std::errc{} || result.ptr != m_buffer.data()+m_buffer.size() || value < m_signedMinimum || value > m_signedMaximum)
            return reject("Signed integer is outside the allowed range.");
        m_committedSigned = value;
        m_hasCommittedSigned = true;
        m_committedText = m_normalizedText = std::to_string(value);
        break;
    }
	case TextEntryMode::UnsignedInteger: {
		unsigned int value = 0;

		if (m_buffer.empty()) {
			if (!m_hasUnsignedDefault) {
				return reject("An unsigned-integer value is required.");
			}

			value = m_unsignedDefault;
		}
		else if (!tryParseUnsigned(m_buffer, value)) {
			return reject("Unsigned integer overflow was prevented.");
		}

		if (value < m_unsignedMinimum ||
			value > m_unsignedMaximum) {

			return reject("Unsigned integer is outside the allowed range.");
		}

		m_committedUnsigned = value;
		m_hasCommittedUnsigned = true;
		m_committedText = std::to_string(value);
		m_normalizedText = m_committedText;
		break;
	}

	case TextEntryMode::AssetName: {
		const std::string trimmed = trimAsciiSpaces(m_buffer);
		if (trimmed.empty()) {
			return reject("Asset name cannot be empty.");
		}

		m_committedText = trimmed;
		m_normalizedText = normalizeAssetName(trimmed);
		break;
	}

	case TextEntryMode::GeneralText:
		m_committedText = m_buffer;
		m_normalizedText = m_buffer;
		break;

	default:
	case TextEntryMode::None:
		return TextEntryAction::None;
	}

	clearActiveSession();
	m_statusMessage = "Entry committed.";
	return TextEntryAction::Committed;
}

TextEntryAction TextEntrySession::reject(const char* message) {
	m_statusMessage = message ? message : "Entry rejected.";
	return TextEntryAction::Rejected;
}

bool TextEntrySession::tryGetCommittedUnsigned(unsigned int& value) const {
	if (!m_hasCommittedUnsigned) {
		return false;
	}

	value = m_committedUnsigned;
	return true;
}

void TextEntrySession::cancel() {
	if (!m_active) {
		return;
	}

	clearCommittedResult();
	clearActiveSession();
	m_statusMessage = "Entry cancelled.";
}

void TextEntrySession::reset() {
	m_active = false;
	m_mode = TextEntryMode::None;
	m_prompt.clear();
	m_buffer.clear();
	m_statusMessage.clear();

	m_maximumLength = 0;
	m_unsignedMinimum = 0;
	m_unsignedMaximum = 0;
	m_unsignedDefault = 0;
	m_hasUnsignedDefault = false;

	clearCommittedResult();
}

void TextEntrySession::clearActiveSession() {
	m_active = false;
	m_mode = TextEntryMode::None;
	m_prompt.clear();
	m_buffer.clear();
	m_maximumLength = 0;
	m_unsignedMinimum = 0;
	m_unsignedMaximum = 0;
	m_unsignedDefault = 0;
	m_hasUnsignedDefault = false;
}

void TextEntrySession::clearCommittedResult() {
    m_replaceNumeric = false;
    m_hasCommittedReal = m_hasCommittedSigned = false;
	m_committedText.clear();
	m_normalizedText.clear();
	m_committedUnsigned = 0;
	m_hasCommittedUnsigned = false;
}

bool TextEntrySession::tryParseUnsigned(
	const std::string& text,
	unsigned int& value) {

	if (text.empty()) {
		return false;
	}

	const unsigned int maximum =
		std::numeric_limits<unsigned int>::max();

	unsigned int parsed = 0;
	for (char character : text) {
		if (character < '0' || character > '9') {
			return false;
		}

		const unsigned int digit =
			static_cast<unsigned int>(character - '0');

		if (parsed > (maximum - digit) / 10u) {
			return false;
		}

		parsed = parsed * 10u + digit;
	}

	value = parsed;
	return true;
}

bool TextEntrySession::isPrintableAscii(unsigned char rawKey) {
	return rawKey >= 32 && rawKey <= 126;
}

bool TextEntrySession::isAssetNameCharacter(unsigned char rawKey) {
	const bool isUppercase = rawKey >= 'A' && rawKey <= 'Z';
	const bool isLowercase = rawKey >= 'a' && rawKey <= 'z';
	const bool isDigit = rawKey >= '0' && rawKey <= '9';

	return isUppercase ||
		isLowercase ||
		isDigit ||
		rawKey == ' ' ||
		rawKey == '_' ||
		rawKey == '-';
}

bool TextEntrySession::isWindowsReservedFilenameCharacter(
	unsigned char rawKey) {

	switch (rawKey) {
	case '<':
	case '>':
	case ':':
	case '"':
	case '/':
	case '\\':
	case '|':
	case '?':
	case '*':
		return true;

	default:
		return false;
	}
}

bool TextEntrySession::isValidInitialText(
	TextEntryMode mode,
	const std::string& value) {

	for (unsigned char character : value) {
		if (mode == TextEntryMode::AssetName) {
			if (!isAssetNameCharacter(character)) {
				return false;
			}
		}
		else if (mode == TextEntryMode::GeneralText) {
			if (!isPrintableAscii(character)) {
				return false;
			}
		}
	}

	return true;
}

std::string TextEntrySession::trimAsciiSpaces(const std::string& value) {
	std::size_t first = 0;
	while (first < value.size() && value[first] == ' ') {
		++first;
	}

	std::size_t last = value.size();
	while (last > first && value[last - 1] == ' ') {
		--last;
	}

	return value.substr(first, last - first);
}

std::string TextEntrySession::normalizeAssetName(
	const std::string& value) {

	std::string normalized = value;
	for (char& character : normalized) {
		if (character == ' ') {
			character = '_';
		}
	}

	return normalized;
}

std::size_t TextEntrySession::decimalDigitCount(unsigned int value) {
	std::size_t digits = 1;
	while (value >= 10u) {
		value /= 10u;
		++digits;
	}

	return digits;
}

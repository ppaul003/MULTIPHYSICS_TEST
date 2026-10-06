#ifndef TEXT_ENTRY_H
#define TEXT_ENTRY_H

#include <cstddef>
#include <string>

enum class TextEntryMode {
	None = 0,
	UnsignedInteger,
	AssetName,
	GeneralText
};

enum class TextEntryAction {
	None = 0,
	Changed,
	Committed,
	Cancelled,
	Rejected
};

class TextEntrySession {
public:
	TextEntrySession() = default;

	bool beginUnsignedInteger(
		const std::string& prompt,
		unsigned int minimum,
		unsigned int maximum,
		unsigned int initialValue
	);

	// Starts with no default. Committing an empty buffer is rejected.
	bool beginUnsignedInteger(
		const std::string& prompt,
		unsigned int minimum,
		unsigned int maximum
	);

	bool beginAssetName(
		const std::string& prompt,
		const std::string& initialValue,
		std::size_t maximumLength
	);

	bool beginGeneralText(
		const std::string& prompt,
		const std::string& initialValue,
		std::size_t maximumLength
	);

	TextEntryAction handleRawKey(unsigned char rawKey);

	bool isActive() const { return m_active; }
	TextEntryMode getMode() const { return m_mode; }

	const std::string& getPrompt() const { return m_prompt; }
	const std::string& getBuffer() const { return m_buffer; }
	const std::string& getCommittedText() const { return m_committedText; }
	const std::string& getNormalizedText() const { return m_normalizedText; }
	const std::string& getStatusMessage() const { return m_statusMessage; }

	bool tryGetCommittedUnsigned(unsigned int& value) const;

	void cancel();
	void reset();

private:
	bool beginUnsignedIntegerSession(
		const std::string& prompt,
		unsigned int minimum,
		unsigned int maximum,
		bool hasDefault,
		unsigned int initValue
	);

	bool beginTextSession(
		TextEntryMode mode,
		const std::string& prompt,
		const std::string& initialValue,
		std::size_t maximumLength
	);

	TextEntryAction handleUnsignedIntegerKey(unsigned char rawKey);
	TextEntryAction handleAssetNameKey(unsigned char rawKey);
	TextEntryAction handleGeneralTextKey(unsigned char rawKey);
	TextEntryAction commitActiveSession();
	TextEntryAction reject(const char* message);

	void clearActiveSession();
	void clearCommittedResult();

	static bool tryParseUnsigned(
		const std::string& text,
		unsigned int& value
	);
	static bool isPrintableAscii(unsigned char rawKey);
	static bool isAssetNameCharacter(unsigned char rawKey);
	static bool isWindowsReservedFilenameCharacter(unsigned char rawKey);
	static bool isValidInitialText(
		TextEntryMode mode,
		const std::string& value
	);
	static std::string trimAsciiSpaces(const std::string& value);
	static std::string normalizeAssetName(const std::string& value);
	static std::size_t decimalDigitCount(unsigned int value);

	bool m_active = false;
	TextEntryMode m_mode = TextEntryMode::None;

	std::string m_prompt;
	std::string m_buffer;
	std::string m_committedText;
	std::string m_normalizedText;
	std::string m_statusMessage;

	std::size_t m_maximumLength = 0;
	unsigned int m_unsignedMinimum = 0;

	unsigned int m_unsignedMaximum = 0;
	unsigned int m_unsignedDefault = 0;
	unsigned int m_committedUnsigned = 0;

	bool m_hasUnsignedDefault = false;
	bool m_hasCommittedUnsigned = false;
};

#endif

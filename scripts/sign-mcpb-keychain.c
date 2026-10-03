/* Detached CMS signing using an exact Keychain identity; private keys stay in Keychain. */
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <Security/CMSEncoder.h>
#include <CommonCrypto/CommonDigest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(OSStatus status, const char *step) {
    if (status != errSecSuccess) {
        fprintf(stderr, "%s failed: OSStatus %d\n", step, (int)status);
        exit(1);
    }
}
int main(int argc, char **argv) {
    if (argc != 4 || strlen(argv[1]) != 40) {
        fprintf(stderr, "Usage: signer CERTIFICATE_SHA1 input.mcpb output.der\n"); return 2;
    }
    const void *keys[] = {kSecClass, kSecReturnRef, kSecMatchLimit};
    const void *values[] = {kSecClassIdentity, kCFBooleanTrue, kSecMatchLimitAll};
    CFDictionaryRef query = CFDictionaryCreate(NULL, keys, values, 3, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFTypeRef matches = NULL;
    check(SecItemCopyMatching(query, &matches), "Identity lookup");
    SecIdentityRef selected = NULL;
    CFArrayRef identities = (CFArrayRef)matches;
    for (CFIndex i = 0; i < CFArrayGetCount(identities); ++i) {
        SecIdentityRef identity = (SecIdentityRef)CFArrayGetValueAtIndex(identities, i);
        SecCertificateRef certificate = NULL;
        check(SecIdentityCopyCertificate(identity, &certificate), "Certificate lookup");
        CFDataRef der = SecCertificateCopyData(certificate);
        unsigned char digest[CC_SHA1_DIGEST_LENGTH];
        CC_SHA1(CFDataGetBytePtr(der), (CC_LONG)CFDataGetLength(der), digest);
        char hex[41];
        for (int n = 0; n < CC_SHA1_DIGEST_LENGTH; ++n) sprintf(hex + 2*n, "%02X", digest[n]);
        if (strcasecmp(hex, argv[1]) == 0) selected = identity;
        CFRelease(der); CFRelease(certificate);
        if (selected) break;
    }
    if (!selected) { fprintf(stderr, "Exact requested identity not found\n"); return 1; }
    CMSEncoderRef encoder = NULL;
    check(CMSEncoderCreate(&encoder), "CMS create");
    check(CMSEncoderSetSignerAlgorithm(encoder, kCMSEncoderDigestAlgorithmSHA256), "SHA256 algorithm");
    check(CMSEncoderAddSigners(encoder, selected), "Exact signer selection");
    check(CMSEncoderSetHasDetachedContent(encoder, true), "Detached content");
    FILE *input = fopen(argv[2], "rb");
    if (!input) { perror("input"); return 1; }
    unsigned char buffer[65536]; size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), input)) > 0)
        check(CMSEncoderUpdateContent(encoder, buffer, count), "CMS content update");
    if (ferror(input)) { fprintf(stderr, "Input read failed\n"); return 1; }
    fclose(input);
    CFDataRef encoded = NULL;
    check(CMSEncoderCopyEncodedContent(encoder, &encoded), "Keychain CMS signing");
    FILE *output = fopen(argv[3], "wb");
    if (!output) { perror("output"); return 1; }
    if (fwrite(CFDataGetBytePtr(encoded), 1, CFDataGetLength(encoded), output) != (size_t)CFDataGetLength(encoded)) {
        fprintf(stderr, "Output write failed\n"); return 1;
    }
    fclose(output);
    CFRelease(encoded); CFRelease(encoder); CFRelease(matches); CFRelease(query);
    puts("Detached SHA256 CMS signature created with the exact Keychain identity.");
    return 0;
}
